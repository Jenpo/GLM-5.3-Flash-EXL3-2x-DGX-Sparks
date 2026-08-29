#!/usr/bin/env python3
"""Backport vLLM's XGrammar speculative-batch termination fix.

The base image pins vLLM ``487ecf187``, whose XGrammar backend can continue
feeding tokens to a matcher after an EOS/stop token terminates it.  That is
especially visible with a multi-token speculative batch: the first trailing
token is rejected, subsequent advances can warn, and the cached termination
flag can become inconsistent with the matcher.

This is a source-exact behavioral backport of upstream vLLM PR #52805, merged
as commit 12f64b39d29282437e35be9aa5db432fb2a1a6e6:

https://github.com/vllm-project/vllm/pull/52805
https://github.com/vllm-project/vllm/commit/12f64b39d29282437e35be9aa5db432fb2a1a6e6

Only the three upstream method edits are made.  The patch is idempotent and
fails closed before writing if the pinned anchors drift or a partial patch is
found.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_XGRAMMAR_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
        "backend_xgrammar.py",
    )
)
MARK = (
    "    # [glm53-xgrammar-termination] Source-exact vLLM 12f64b39 backport.\n"
)

ACCEPT_OLD = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
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
'''

ACCEPT_UPSTREAM = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
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
            if self._is_terminated:
                break
        return True
'''

VALIDATE_OLD = '''    def validate_tokens(self, tokens: list[int]) -> list[int]:
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
'''

VALIDATE_UPSTREAM = '''    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

RESET_OLD = '''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''

RESET_UPSTREAM = '''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False
'''


def counts(text: str) -> tuple[list[int], list[int]]:
    old = [text.count(x) for x in (ACCEPT_OLD, VALIDATE_OLD, RESET_OLD)]
    new = [
        text.count(ACCEPT_UPSTREAM),
        text.count(VALIDATE_UPSTREAM),
        text.count(RESET_UPSTREAM),
    ]
    return old, new


def verified_upstream_state(text: str) -> bool:
    old, new = counts(text)
    return old == [0, 0, 0] and new == [1, 1, 1]


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")

    source = TARGET.read_text()
    old, new = counts(source)
    marker_count = source.count(MARK)

    if marker_count:
        if marker_count != 1 or not verified_upstream_state(source):
            raise SystemExit(
                f"{TARGET}: partial/inconsistent xgrammar termination patch "
                f"(marker={marker_count}, old={old}, new={new})"
            )
        compile(source, str(TARGET), "exec")
        print(f"{TARGET.name}: xgrammar termination patch already present")
        return 0

    # A newer image may already contain the exact merged upstream behavior.
    # Accept that exact state without adding a recipe marker; any other drift
    # remains fatal.
    if verified_upstream_state(source):
        compile(source, str(TARGET), "exec")
        print(f"{TARGET.name}: upstream xgrammar termination fix already present")
        return 0

    if old != [1, 1, 1] or new != [0, 0, 0]:
        raise SystemExit(
            f"{TARGET}: pinned xgrammar anchors drifted; refusing partial write "
            f"(old={old}, new={new})"
        )

    patched = source.replace(ACCEPT_OLD, MARK + ACCEPT_UPSTREAM, 1)
    patched = patched.replace(VALIDATE_OLD, VALIDATE_UPSTREAM, 1)
    patched = patched.replace(RESET_OLD, RESET_UPSTREAM, 1)
    if not verified_upstream_state(patched) or patched.count(MARK) != 1:
        raise SystemExit(f"{TARGET}: post-patch verification failed")
    compile(patched, str(TARGET), "exec")

    tmp = TARGET.with_name(f".{TARGET.name}.glm53-xgrammar.tmp")
    try:
        tmp.write_text(patched)
        os.chmod(tmp, stat.S_IMODE(TARGET.stat().st_mode))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(
        f"patched {TARGET.name} "
        "(vLLM #52805: stop XGrammar token batches at termination)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
