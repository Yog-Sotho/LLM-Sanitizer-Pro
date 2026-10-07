"""Chat-dataset validation for conversational fine-tuning data.

Fine-tuning jobs fail — or silently train on garbage — because of structural
problems in ``messages``-format datasets: broken role alternation, empty
turns, conversations with no assistant reply to learn from, or single
examples that blow past the model's context window. This module lints each
record against those rules *before* the data reaches a trainer.

Formats: OpenAI ``messages`` (string content or content parts such as
``[{"type": "text", ...}, {"type": "image_url", ...}]``, assistant
``tool_calls`` and ``tool`` results) and ShareGPT ``conversations``
(``[{"from": "human", "value": ...}]``).

Strict rules (default):
  * the conversation is a non-empty list of ``{"role": str, "content": ...}``
  * roles are in the allowed set (default: system/user/assistant; add
    ``tool`` with --chat-roles for tool-use data)
  * at most one system message, and only at position 0
  * the first non-system message is from the user
  * user/assistant turns strictly alternate; an assistant turn may contain
    tool round-trips (assistant tool_calls -> tool results -> assistant)
  * the conversation ends on an assistant message (the training target)
  * no empty/whitespace-only content (an assistant turn that only calls
    tools may have none)
  * tool calls name a function and carry JSON arguments; every tool result
    answers an open call (by tool_call_id) and every call is answered
  * optional token budget over the whole conversation, rendered through the
    tokenizer's chat template when it has one

Lenient mode keeps the structural and tool-call checks but drops the
ordering/alternation requirements — useful for multi-agent traces.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    return make_counters(tokenizer_name)[0]


def make_counters(tokenizer_name: str = 'whitespace'
                  ) -> Tuple[Callable[[str], int],
                             Optional[Callable[[List[Dict[str, Any]]], int]]]:
    """(text counter, conversation counter). The conversation counter renders
    messages with the tokenizer's chat template — role markup and special
    tokens included, as the trainer sees them — and is None for whitespace
    counting or tokenizers without a template."""
    if tokenizer_name == 'whitespace':
        return (lambda text: len(text.split())), None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        logging.warning(f"Could not load tokenizer '{tokenizer_name}' for chat "
                        f"validation ({exc}); counting whitespace tokens instead.")
        return (lambda text: len(text.split())), None

    def count_text(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    if not getattr(tok, 'chat_template', None):
        return count_text, None

    def count_conversation(messages: List[Dict[str, Any]]) -> int:
        ids = tok.apply_chat_template(messages, tokenize=True)
        if hasattr(ids, 'keys'):  # BatchEncoding in newer transformers releases
            ids = ids['input_ids']
        return len(ids)

    return count_text, count_conversation


# OpenAI-format extensions
BAD_TOOL_CALL = 'bad_tool_call'
TOOL_ARGS_NOT_JSON = 'tool_arguments_not_json'
ORPHAN_TOOL_RESULT = 'orphan_tool_result'
UNANSWERED_TOOL_CALL = 'unanswered_tool_call'

# ShareGPT 'from' values -> OpenAI roles
_SHAREGPT_ROLES = {
    'human': 'user', 'user': 'user', 'gpt': 'assistant', 'assistant': 'assistant',
    'chatgpt': 'assistant', 'bard': 'assistant', 'model': 'assistant', 'system': 'system',
    'tool': 'tool', 'observation': 'tool', 'function_response': 'tool',
    'function_call': 'assistant',
}


def is_sharegpt(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(m, dict) and 'from' in m and 'value' in m for m in value))


def sharegpt_to_messages(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[{'from': 'human', 'value': ...}] -> [{'role': 'user', 'content': ...}]."""
    out = []
    for turn in conversation:
        speaker = str(turn.get('from', '')).strip().lower()
        out.append({'role': _SHAREGPT_ROLES.get(speaker, speaker), 'content': turn.get('value')})
    return out


def conversation_of(record: Dict[str, Any]) -> Optional[Any]:
    """A record's messages: OpenAI 'messages', or ShareGPT 'conversations'."""
    if 'messages' in record:
        msgs = record['messages']
        return sharegpt_to_messages(msgs) if is_sharegpt(msgs) else msgs
    if is_sharegpt(record.get('conversations')):
        return sharegpt_to_messages(record['conversations'])
    return None


def _content_text(content: Any) -> Optional[str]:
    """Text of a message's content: a string, or OpenAI content parts
    ([{'type': 'text', 'text': ...}, {'type': 'image_url', ...}]). None when
    the shape is invalid."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and 'type' in p for p in content):
        texts = [p.get('text') for p in content if p.get('type') == 'text']
        if any(not isinstance(t, str) for t in texts):
            return None
        # A non-text part (image, audio, file) is content even without text.
        has_media = any(p.get('type') != 'text' for p in content)
        joined = ' '.join(t for t in texts if isinstance(t, str))
        return joined if joined.strip() or not has_media else '<media>'
    return None


def _tool_calls(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    calls = message.get('tool_calls')
    return calls if isinstance(calls, list) and calls else None


class ChatValidator:
    """Validates chat records and tallies failure reasons.

    Accepts OpenAI chat format (string or content-part messages, assistant
    tool_calls, tool results) and ShareGPT 'conversations'. In strict mode a
    conversation must look like: [system] user assistant (user assistant)*,
    where an assistant turn may include tool round-trips: assistant(tool_calls)
    -> tool result(s) -> ... -> final assistant answer."""

    def __init__(self, allowed_roles: Sequence[str] = DEFAULT_ROLES, lenient: bool = False,
                 max_tokens: Optional[int] = None,
                 token_counter: Optional[Callable[[str], int]] = None,
                 conversation_counter: Optional[Callable[[List[Dict[str, Any]]], int]] = None
                 ) -> None:
        self.allowed_roles = {r.strip().lower() for r in allowed_roles if r.strip()}
        self.lenient = lenient
        self.max_tokens = max_tokens
        self._count_tokens = token_counter or (lambda text: len(text.split()))
        self._count_conversation = conversation_counter
        self.reason_counts: Dict[str, int] = {}

    def check(self, record: Dict[str, Any]) -> Optional[str]:
        """Return the first failing reason, or None when the record is valid."""
        reason = self._check(record)
        if reason:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        return reason

    def _check(self, record: Dict[str, Any]) -> Optional[str]:
        messages = conversation_of(record)
        if messages is None:
            return MISSING_MESSAGES
        if not isinstance(messages, list):
            return NOT_A_LIST
        if not messages:
            return EMPTY_CONVERSATION

        roles: List[str] = []
        texts: List[str] = []
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get('role'), str):
                return BAD_MESSAGE_SCHEMA
            role = m['role'].lower()
            if role not in self.allowed_roles:
                return UNKNOWN_ROLE
            calls = _tool_calls(m) if role == 'assistant' else None
            content = m.get('content')
            if calls is not None and content in (None, ''):
                text = ''                          # tool-call turns may carry no text
            else:
                maybe = _content_text(content)
                if maybe is None:
                    return BAD_MESSAGE_SCHEMA
                text = maybe
                if not text.strip() and role != 'tool':  # empty tool output is legitimate
                    return EMPTY_CONTENT
            if calls is not None:
                for call in calls:
                    fn = call.get('function') if isinstance(call, dict) else None
                    if not (isinstance(fn, dict) and isinstance(fn.get('name'), str)
                            and fn['name'].strip()):
                        return BAD_TOOL_CALL
                    args = fn.get('arguments', '{}')
                    if isinstance(args, str):
                        try:
                            json.loads(args or '{}')
                        except ValueError:
                            return TOOL_ARGS_NOT_JSON
                    elif not isinstance(args, dict):
                        return TOOL_ARGS_NOT_JSON
                    text += f" {fn['name']} {args if isinstance(args, str) else json.dumps(args)}"
            roles.append(role)
            texts.append(text)

        if roles.count('system') > 1:
            return MULTIPLE_SYSTEM
        if 'system' in roles and roles.index('system') != 0:
            return SYSTEM_NOT_FIRST
        if 'assistant' not in roles:
            return NO_ASSISTANT
        tool_problem = self._check_tool_links(messages)
        if tool_problem:
            return tool_problem
        if self.max_tokens is not None and self._tokens(messages, texts) > self.max_tokens:
            return TOO_MANY_TOKENS
        if not self.lenient:
            return self._check_order(messages, roles)
        return None

    def _tokens(self, messages: List[Dict[str, Any]], texts: List[str]) -> int:
        if self._count_conversation is not None:
            try:
                return self._count_conversation(messages)
            except Exception as exc:  # template rejects this shape: fall back
                logging.debug(f"chat template counting failed ({exc}); summing contents")
        return sum(self._count_tokens(t) for t in texts)

    @staticmethod
    def _check_tool_links(messages: List[Dict[str, Any]]) -> Optional[str]:
        """Every tool result answers an earlier, still-open call (matched by
        tool_call_id when given, else in order), and every call is answered
        before the next user turn. Conversations without structured
        tool_calls (e.g. ShareGPT function_call/observation text turns) are
        not linked."""
        if not any(_tool_calls(m) for m in messages
                   if str(m.get('role', '')).lower() == 'assistant'):
            return None
        pending: List[Optional[str]] = []
        for m in messages:
            role = str(m.get('role', '')).lower()
            calls = _tool_calls(m) if role == 'assistant' else None
            if calls:
                pending += [c.get('id') if isinstance(c, dict) else None for c in calls]
            elif role == 'tool':
                call_id = m.get('tool_call_id')
                if not pending:
                    return ORPHAN_TOOL_RESULT
                if call_id is None:
                    pending.pop(0)
                elif call_id in pending:
                    pending.remove(call_id)
                else:
                    return ORPHAN_TOOL_RESULT
            elif role == 'user' and pending:
                return UNANSWERED_TOOL_CALL
        return UNANSWERED_TOOL_CALL if pending else None

    @staticmethod
    def _check_order(messages: List[Dict[str, Any]], roles: List[str]) -> Optional[str]:
        start = 1 if roles[0] == 'system' else 0
        # Without structured tool_calls, a tool turn may follow any assistant
        # turn (text-encoded calls, as in ShareGPT function_call/observation).
        structured = any(_tool_calls(m) for m, r in zip(messages, roles) if r == 'assistant')
        state = 'user'            # what the next message must be
        prev = ''
        for i, (m, role) in enumerate(zip(messages[start:], roles[start:])):
            calls = _tool_calls(m) if role == 'assistant' else None
            if role == 'user':
                if state != 'user':
                    return NOT_ALTERNATING
                state = 'assistant'
            elif role == 'assistant':
                if i == 0:
                    return FIRST_NOT_USER
                if state == 'user':
                    return NOT_ALTERNATING
                state = 'tool' if calls else 'user'
            elif role == 'tool':
                legacy = not structured and prev in ('assistant', 'tool')
                if state not in ('tool', 'tool_or_assistant') and not legacy:
                    return NOT_ALTERNATING if i else FIRST_NOT_USER
                state = 'tool_or_assistant'
            else:  # custom roles allowed via --chat-roles do not take part in turns
                continue
            prev = role
        if state != 'user':
            return LAST_NOT_ASSISTANT
        return None
