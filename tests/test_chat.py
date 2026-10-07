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


TOOLS = ('system', 'user', 'assistant', 'tool')


def call(call_id, name="get_weather", arguments='{"city": "Paris"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_conversation(*extra):
    return {"messages": [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": None,
         "tool_calls": [call("c1"), call("c2", arguments='{"city": "Rome"}')]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "24C cloudy"},
        {"role": "assistant", "content": "Paris is 18C and sunny; Rome 24C and cloudy."},
        *extra,
    ]}


class TestToolCalling:
    def test_valid_tool_round_trip(self):
        assert ChatValidator(allowed_roles=TOOLS).check(tool_conversation()) is None

    def test_multi_turn_after_tools(self):
        rec = tool_conversation({"role": "user", "content": "Thanks"},
                                {"role": "assistant", "content": "Anytime."})
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_chained_tool_calls_in_one_turn(self):
        rec = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [call("a")]},
            {"role": "tool", "tool_call_id": "a", "content": "r1"},
            {"role": "assistant", "content": "One more lookup.", "tool_calls": [call("b")]},
            {"role": "tool", "tool_call_id": "b", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]}
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_dict_arguments_accepted(self):
        rec = tool_conversation()
        rec["messages"][1]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_arguments_must_be_json(self):
        rec = tool_conversation()
        rec["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{city: Paris"
        assert ChatValidator(allowed_roles=TOOLS).check(rec) == chat.TOOL_ARGS_NOT_JSON

    def test_call_needs_function_name(self):
        rec = tool_conversation()
        rec["messages"][1]["tool_calls"][0]["function"]["name"] = " "
        assert ChatValidator(allowed_roles=TOOLS).check(rec) == chat.BAD_TOOL_CALL

    def test_orphan_tool_result(self):
        rec = tool_conversation()
        rec["messages"][3]["tool_call_id"] = "nope"
        v = ChatValidator(allowed_roles=TOOLS, lenient=True)
        assert v.check(rec) == chat.ORPHAN_TOOL_RESULT

    def test_unanswered_call(self):
        rec = tool_conversation()
        del rec["messages"][3]
        assert ChatValidator(allowed_roles=TOOLS).check(rec) == chat.UNANSWERED_TOOL_CALL

    def test_unanswered_call_before_user_turn(self):
        rec = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [call("a")]},
            {"role": "user", "content": "hello?"},
            {"role": "assistant", "content": "sorry"},
        ]}
        assert ChatValidator(allowed_roles=TOOLS, lenient=True).check(rec) == \
            chat.UNANSWERED_TOOL_CALL

    def test_results_without_ids_match_in_order(self):
        rec = tool_conversation()
        for m in rec["messages"]:
            m.pop("tool_call_id", None)
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_conversation_cannot_end_on_tool_call(self):
        rec = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [call("a")]},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
        ]}
        assert ChatValidator(allowed_roles=TOOLS).check(rec) == chat.LAST_NOT_ASSISTANT

    def test_tool_result_cannot_follow_user(self):
        rec = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [call("a")]},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "q2"},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
            {"role": "assistant", "content": "y"},
        ]}
        assert ChatValidator(allowed_roles=TOOLS).check(rec) == chat.ORPHAN_TOOL_RESULT

    def test_empty_tool_output_allowed(self):
        rec = tool_conversation()
        rec["messages"][2]["content"] = ""
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_tool_call_arguments_count_toward_budget(self):
        rec = tool_conversation()
        words = sum(len(str(m.get("content") or "").split()) for m in rec["messages"])
        assert ChatValidator(allowed_roles=TOOLS, max_tokens=words).check(rec) == \
            chat.TOO_MANY_TOKENS


class TestContentParts:
    def test_text_parts(self):
        rec = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Describe"},
                                         {"type": "image_url", "image_url": {"url": "x.png"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
        ]}
        assert ChatValidator().check(rec) is None

    def test_image_only_user_turn_is_content(self):
        rec = {"messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x.png"}}]},
            {"role": "assistant", "content": "A cat."},
        ]}
        assert ChatValidator().check(rec) is None

    def test_empty_text_parts_rejected(self):
        rec = {"messages": [{"role": "user", "content": [{"type": "text", "text": " "}]},
                            {"role": "assistant", "content": "a"}]}
        assert ChatValidator().check(rec) == chat.EMPTY_CONTENT

    def test_malformed_parts_rejected(self):
        for content in ([{"text": "no type"}], [{"type": "text", "text": 3}], ["raw"], 42):
            rec = {"messages": [{"role": "user", "content": content},
                                {"role": "assistant", "content": "a"}]}
            assert ChatValidator().check(rec) == chat.BAD_MESSAGE_SCHEMA, content


class TestShareGPT:
    CONV = {"conversations": [{"from": "system", "value": "Be brief."},
                              {"from": "human", "value": "Hi"},
                              {"from": "gpt", "value": "Hello!"}]}

    def test_converts_roles(self):
        assert chat.sharegpt_to_messages(self.CONV["conversations"]) == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}]

    def test_validates_conversations_field(self):
        assert ChatValidator().check(self.CONV) is None

    def test_validates_sharegpt_items_under_messages(self):
        assert ChatValidator().check({"messages": self.CONV["conversations"]}) is None

    def test_invalid_sharegpt_order(self):
        rec = {"conversations": [{"from": "gpt", "value": "a"}, {"from": "human", "value": "q"}]}
        assert ChatValidator().check(rec) == chat.FIRST_NOT_USER

    def test_function_call_observation_turns(self):
        rec = {"conversations": [{"from": "human", "value": "weather?"},
                                 {"from": "function_call", "value": '{"name": "w"}'},
                                 {"from": "observation", "value": "sunny"},
                                 {"from": "gpt", "value": "It is sunny."}]}
        assert ChatValidator(allowed_roles=TOOLS).check(rec) is None

    def test_unknown_speaker_rejected(self):
        rec = {"conversations": [{"from": "narrator", "value": "x"},
                                 {"from": "gpt", "value": "a"}]}
        assert ChatValidator().check(rec) == chat.UNKNOWN_ROLE

    def test_not_sharegpt_without_messages(self):
        assert ChatValidator().check({"conversations": "nope"}) == chat.MISSING_MESSAGES

    def test_format_chatml_converts(self):
        from sanitizer_pro.core import format_chatml
        assert format_chatml(self.CONV)["messages"][1] == {"role": "user", "content": "Hi"}


class TestChatTemplateBudget:
    def test_conversation_counter_preferred(self):
        seen = []

        def counter(messages):
            seen.append(messages)
            return 100

        v = ChatValidator(max_tokens=50, conversation_counter=counter)
        assert v.check(VALID) == chat.TOO_MANY_TOKENS
        assert seen and seen[0][0]["role"] == "system"

    def test_falls_back_when_template_fails(self):
        def counter(messages):
            raise ValueError("template rejects tool role")

        v = ChatValidator(max_tokens=50, conversation_counter=counter)
        assert v.check(VALID) is None

    def test_whitespace_has_no_conversation_counter(self):
        text, conv = chat.make_counters('whitespace')
        assert text("a b c") == 3 and conv is None
