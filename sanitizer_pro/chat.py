"""Chat-dataset validation for conversational fine-tuning data.

Fine-tuning jobs fail — or silently train on garbage — because of structural
problems in ``messages``-format datasets: broken role alternation, empty
turns, conversations with no assistant reply to learn from, or single
examples that blow past the model's context window. This module lints each
record against those rules *before* the data reaches a trainer.

Strict rules (default):
  * ``messages`` is a non-empty list of ``{"role": str, "content": str}``
  * roles are in the allowed set (default: system/user/assistant)
  * at most one system message, and only at position 0
  * the first non-system message is from the user
  * user/assistant turns strictly alternate
  * the conversation ends on an assistant message (the training target)
  * no empty/whitespace-only content
  * optional token budget over the whole conversation

Lenient mode keeps the structural checks (schema, known roles, non-empty
content, assistant present) but drops the ordering/alternation requirements —
useful for multi-agent or tool-augmented traces.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_ROLES = ('system', 'user', 'assistant')

# Failure reason identifiers (stable: they appear in stats files)
MISSING_MESSAGES = 'missing_messages'
NOT_A_LIST = 'messages_not_a_list'
EMPTY_CONVERSATION = 'empty_conversation'
BAD_MESSAGE_SCHEMA = 'bad_message_schema'
UNKNOWN_ROLE = 'unknown_role'
EMPTY_CONTENT = 'empty_content'
MULTIPLE_SYSTEM = 'multiple_system'
SYSTEM_NOT_FIRST = 'system_not_first'
FIRST_NOT_USER = 'first_not_user'
NO_ASSISTANT = 'no_assistant_reply'
NOT_ALTERNATING = 'roles_not_alternating'
LAST_NOT_ASSISTANT = 'last_not_assistant'
TOO_MANY_TOKENS = 'too_many_tokens'


def make_token_counter(tokenizer_name: str = 'whitespace') -> Callable[[str], int]:
    """Return a text→token-count callable (HF tokenizer or whitespace)."""
    if tokenizer_name != 'whitespace':
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except Exception as exc:
            logging.warning(f"Could not load tokenizer '{tokenizer_name}' for chat "
                            f"validation ({exc}); counting whitespace tokens instead.")
    return lambda text: len(text.split())


class ChatValidator:
    """Validates messages-format records and tallies failure reasons."""

    def __init__(self, allowed_roles: Sequence[str] = DEFAULT_ROLES, lenient: bool = False,
                 max_tokens: Optional[int] = None,
                 token_counter: Optional[Callable[[str], int]] = None) -> None:
        self.allowed_roles = {r.strip().lower() for r in allowed_roles if r.strip()}
        self.lenient = lenient
        self.max_tokens = max_tokens
        self._count_tokens = token_counter or (lambda text: len(text.split()))
        self.reason_counts: Dict[str, int] = {}

    def check(self, record: Dict[str, Any]) -> Optional[str]:
        """Return the first failing reason, or None when the record is valid."""
        reason = self._check(record)
        if reason:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        return reason

    def _check(self, record: Dict[str, Any]) -> Optional[str]:
        if 'messages' not in record:
            return MISSING_MESSAGES
        messages = record['messages']
        if not isinstance(messages, list):
            return NOT_A_LIST
        if not messages:
            return EMPTY_CONVERSATION

        roles: List[str] = []
        total_tokens = 0
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get('role'), str):
                return BAD_MESSAGE_SCHEMA
            content = m.get('content')
            if not isinstance(content, str):
                return BAD_MESSAGE_SCHEMA
            role = m['role'].lower()
            if role not in self.allowed_roles:
                return UNKNOWN_ROLE
            if not content.strip():
                return EMPTY_CONTENT
            roles.append(role)
            if self.max_tokens is not None:
                total_tokens += self._count_tokens(content)

        if roles.count('system') > 1:
            return MULTIPLE_SYSTEM
        if 'system' in roles and roles.index('system') != 0:
            return SYSTEM_NOT_FIRST
        if 'assistant' not in roles:
            return NO_ASSISTANT
        if self.max_tokens is not None and total_tokens > self.max_tokens:
            return TOO_MANY_TOKENS

        if not self.lenient:
            turns = roles[1:] if roles[0] == 'system' else roles
            core = [r for r in turns if r in ('user', 'assistant')]
            if not core or core[0] != 'user':
                return FIRST_NOT_USER
            for prev, cur in zip(core, core[1:]):
                if prev == cur:
                    return NOT_ALTERNATING
            if turns[-1] != 'assistant':
                return LAST_NOT_ASSISTANT
        return None
