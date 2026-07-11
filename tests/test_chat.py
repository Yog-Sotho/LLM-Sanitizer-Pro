"""Tests for chat-dataset validation."""
from sanitizer_pro import chat
from sanitizer_pro.chat import ChatValidator, make_token_counter


def msgs(*pairs):
    return {"messages": [{"role": r, "content": c} for r, c in pairs]}


VALID = msgs(("system", "Be helpful."), ("user", "Hi there"), ("assistant", "Hello!"))


class TestStrictValidation:
    def test_valid_conversation(self):
        assert ChatValidator().check(VALID) is None

    def test_valid_without_system(self):
        assert ChatValidator().check(msgs(("user", "Hi"), ("assistant", "Hello!"))) is None

    def test_valid_multi_turn(self):
        rec = msgs(("user", "Q1"), ("assistant", "A1"), ("user", "Q2"), ("assistant", "A2"))
        assert ChatValidator().check(rec) is None

    def test_missing_messages(self):
        assert ChatValidator().check({"text": "hi"}) == chat.MISSING_MESSAGES

    def test_messages_not_a_list(self):
        assert ChatValidator().check({"messages": "hi"}) == chat.NOT_A_LIST

    def test_empty_conversation(self):
        assert ChatValidator().check({"messages": []}) == chat.EMPTY_CONVERSATION

    def test_bad_message_schema(self):
        assert ChatValidator().check({"messages": ["hi"]}) == chat.BAD_MESSAGE_SCHEMA
        assert ChatValidator().check(
            {"messages": [{"role": "user", "content": 42}]}) == chat.BAD_MESSAGE_SCHEMA

    def test_unknown_role(self):
        assert ChatValidator().check(msgs(("narrator", "Once"))) == chat.UNKNOWN_ROLE

    def test_empty_content(self):
        rec = msgs(("user", "Hi"), ("assistant", "   "))
        assert ChatValidator().check(rec) == chat.EMPTY_CONTENT

    def test_multiple_system(self):
        rec = msgs(("system", "a"), ("user", "q"), ("system", "b"), ("assistant", "r"))
        assert ChatValidator().check(rec) == chat.MULTIPLE_SYSTEM

    def test_system_not_first(self):
        rec = msgs(("user", "q"), ("system", "late"), ("assistant", "r"))
        assert ChatValidator().check(rec) == chat.SYSTEM_NOT_FIRST

    def test_no_assistant_reply(self):
        assert ChatValidator().check(msgs(("user", "anyone?"))) == chat.NO_ASSISTANT

    def test_first_not_user(self):
        rec = msgs(("assistant", "unprompted"), ("user", "ok"), ("assistant", "done"))
        assert ChatValidator().check(rec) == chat.FIRST_NOT_USER

    def test_roles_not_alternating(self):
        rec = msgs(("user", "q1"), ("user", "q2"), ("assistant", "a"))
        assert ChatValidator().check(rec) == chat.NOT_ALTERNATING

    def test_last_not_assistant(self):
        rec = msgs(("user", "q"), ("assistant", "a"), ("user", "dangling"))
        assert ChatValidator().check(rec) == chat.LAST_NOT_ASSISTANT

    def test_reason_counting(self):
        v = ChatValidator()
        v.check(msgs(("user", "anyone?")))
        v.check(msgs(("user", "anyone else?")))
        v.check(VALID)
        assert v.reason_counts == {chat.NO_ASSISTANT: 2}


class TestLenientMode:
    def test_ordering_rules_skipped(self):
        rec = msgs(("user", "q1"), ("user", "q2"), ("assistant", "a"), ("user", "dangling"))
        assert ChatValidator(lenient=True).check(rec) is None

    def test_structural_rules_still_apply(self):
        assert ChatValidator(lenient=True).check(msgs(("user", "no reply"))) == chat.NO_ASSISTANT
        assert ChatValidator(lenient=True).check(msgs(("wizard", "x"))) == chat.UNKNOWN_ROLE


class TestCustomRoles:
    def test_tool_role_allowed(self):
        rec = msgs(("user", "look this up"), ("assistant", "calling tool"),
                   ("tool", "result: 42"), ("assistant", "It is 42."))
        v = ChatValidator(allowed_roles=('system', 'user', 'assistant', 'tool'), lenient=True)
        assert v.check(rec) is None

    def test_tool_role_rejected_by_default(self):
        rec = msgs(("user", "q"), ("tool", "data"), ("assistant", "a"))
        assert ChatValidator().check(rec) == chat.UNKNOWN_ROLE


class TestTokenBudget:
    def test_within_budget(self):
        v = ChatValidator(max_tokens=10)
        assert v.check(msgs(("user", "short question"), ("assistant", "short answer"))) is None

    def test_over_budget(self):
        v = ChatValidator(max_tokens=5)
        rec = msgs(("user", "one two three four"), ("assistant", "five six seven"))
        assert v.check(rec) == chat.TOO_MANY_TOKENS

    def test_no_budget_no_check(self):
        long = msgs(("user", "word " * 5000), ("assistant", "ok then"))
        assert ChatValidator().check(long) is None


def test_whitespace_token_counter():
    counter = make_token_counter('whitespace')
    assert counter("one two three") == 3
