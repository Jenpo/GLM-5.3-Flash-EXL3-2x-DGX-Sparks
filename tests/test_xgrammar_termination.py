#!/usr/bin/env python3
"""Regression tests for the vLLM #52805 XGrammar runtime backport."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_xgrammar_termination.py",
        ROOT / "overlay" / "patch_xgrammar_termination.py",
    )
    if p.is_file()
)
INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
    "backend_xgrammar.py"
)
MARK = "# [glm53-xgrammar-termination] Source-exact vLLM 12f64b39 backport."

# Exact vLLM 487ecf187 method bodies, embedded in a dependency-free harness.
PINNED_FIXTURE = '''class _Logger:
    def error(self, *args):
        pass


logger = _Logger()


class XgrammarGrammar:
    def __init__(self, matcher):
        self.matcher = matcher
        self.num_processed_tokens = 0
        self._is_terminated = False

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''


class FakeMatcher:
    """Small matcher that terminates on token 99 and detects over-advance."""

    def __init__(self):
        self.accepted: list[int] = []
        self.calls_after_termination = 0

    def accept_token(self, token: int) -> bool:
        if self.is_terminated():
            self.calls_after_termination += 1
            return False
        if token == -1:
            return False
        self.accepted.append(token)
        return True

    def is_terminated(self) -> bool:
        return bool(self.accepted and self.accepted[-1] == 99)

    def rollback(self, count: int) -> None:
        if count:
            del self.accepted[-count:]

    def reset(self) -> None:
        self.accepted.clear()

    def fill_next_token_bitmask(self, bitmask, idx: int) -> None:
        pass


def run_patch(path: Path, *, ok: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_XGRAMMAR_BACKEND_PY"] = str(path)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def assert_behavior(source: str) -> None:
    namespace: dict[str, object] = {}
    exec(compile(source, "patched_backend_fixture.py", "exec"), namespace)
    grammar_cls = namespace["XgrammarGrammar"]

    matcher = FakeMatcher()
    grammar = grammar_cls(matcher)
    assert grammar.accept_tokens("req", [7, 99, 8])
    assert matcher.accepted == [7, 99]
    assert matcher.calls_after_termination == 0
    assert grammar.num_processed_tokens == 2
    assert grammar.is_terminated()
    assert grammar.accept_tokens("req", [8])
    assert matcher.accepted == [7, 99]
    assert matcher.calls_after_termination == 0

    grammar.reset()
    assert matcher.accepted == []
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()

    assert grammar.validate_tokens([7, 99, 8]) == [7, 99]
    assert matcher.accepted == []
    assert matcher.calls_after_termination == 0
    assert not grammar.is_terminated()

    assert grammar.accept_tokens("req", [99])
    assert grammar.validate_tokens([8]) == []
    assert matcher.calls_after_termination == 0


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "backend_xgrammar.py"
        target.write_text(PINNED_FIXTURE)
        run_patch(target)
        patched = target.read_text()
        assert patched.count(MARK) == 1
        assert "Tokens after termination are ignored." in patched
        assert "if self.matcher.is_terminated():\n                    break" in patched
        assert "self.matcher.reset()\n        self.num_processed_tokens = 0" in patched
        assert_behavior(patched)

        run_patch(target)
        assert target.read_text() == patched

        # Exact merged behavior is accepted when a newer image already has it.
        target.write_text(patched.replace(f"    {MARK}\n", "", 1))
        upstream = target.read_text()
        run_patch(target)
        assert target.read_text() == upstream


def test_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "backend_xgrammar.py"
        drifted = PINNED_FIXTURE.replace(
            "Returns False if the FSM failed to advance.",
            "Returns False on failure.",
            1,
        )
        target.write_text(drifted)
        run_patch(target, ok=False)
        assert target.read_text() == drifted

        partial = PINNED_FIXTURE.replace(
            "    def accept_tokens",
            f"    {MARK}\n    def accept_tokens",
            1,
        )
        target.write_text(partial)
        run_patch(target, ok=False)
        assert target.read_text() == partial


def test_installed_copy_if_present() -> None:
    source = Path(os.environ.get("GLM53_XGRAMMAR_BACKEND_PY_SRC", INSTALLED))
    if not source.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "backend_xgrammar.py"
        target.write_bytes(source.read_bytes())
        run_patch(target)
        patched = target.read_text()
        compile(patched, str(target), "exec")
        assert "Tokens after termination are ignored." in patched
        assert "self._is_terminated = False" in patched
        run_patch(target)


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-' in launcher
    assert launcher.count("python3 /opt/glm53/patch_xgrammar_termination.py") == 2
    assert (
        "-v '/tmp/patch_xgrammar_termination.py:"
        "/opt/glm53/patch_xgrammar_termination.py:ro'" in launcher
    )
    assert (
        '-v "$XGRAMMAR_PATCH_HOST:'
        '/opt/glm53/patch_xgrammar_termination.py:ro"' in launcher
    )
    assert "scp -q -o BatchMode=yes \"$XGRAMMAR_PATCH_HOST\"" in launcher
    assert "COPY overlay/patch_xgrammar_termination.py" in image
    assert "RUN python3 /opt/glm53/patch_xgrammar_termination.py" in image
    assert "python3 /opt/glm53/test_xgrammar_termination.py" in image


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("xgrammar termination patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
