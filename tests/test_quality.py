"""Tests for quality scoring and content filtering."""
from sanitizer_pro.quality import (
    _check_quality_reason, contains_profanity, extract_text_for_quality, is_code_heuristic,
)
from sanitizer_pro.settings import SanitizerConfig


def make_args(**overrides) -> SanitizerConfig:
    base = dict(min_chars=10, max_chars=1000, min_words=3,
                min_ascii_ratio=0.5, min_unique_ratio=0.2, reject_allcaps=False)
    base.update(overrides)
    return SanitizerConfig(**base)


class TestQualityChecks:
    def test_passes(self):
        assert _check_quality_reason("a normal sentence with enough words", make_args()) is None

    def test_empty(self):
        assert _check_quality_reason("", make_args()) == 'empty text'

    def test_too_short(self):
        assert 'too short' in _check_quality_reason("hi", make_args())

    def test_too_long(self):
        assert 'too long' in _check_quality_reason("x" * 2000, make_args())

    def test_too_few_words(self):
        assert 'too few words' in _check_quality_reason("aaaa bbbbbbbb", make_args())

    def test_low_unique_ratio(self):
        text = "word " * 50
        assert 'unique' in _check_quality_reason(text.strip(), make_args(min_unique_ratio=0.5))

    def test_allcaps(self):
        args = make_args(reject_allcaps=True, allcaps_min_len=10, allcaps_min_alpha=5)
        assert _check_quality_reason("THIS IS ALL SHOUTING TEXT HERE", args) == 'all-caps'


class TestExtractText:
    def test_flat(self):
        assert "hello" in extract_text_for_quality({"text": "hello"})

    def test_nested(self):
        out = extract_text_for_quality({"a": {"b": ["deep text"]}})
        assert "deep text" in out

    def test_text_fields_selection(self):
        out = extract_text_for_quality({"keep": "yes", "skip": "no"}, text_fields=["keep"])
        assert "yes" in out and "no" not in out

    def test_budget_limits_output(self):
        out = extract_text_for_quality({"t": "x" * 100000})
        assert len(out) <= 8192


class TestCodeHeuristic:
    def test_python(self):
        assert is_code_heuristic("def main():\n    import os\n    return os.name")

    def test_javascript(self):
        assert is_code_heuristic("const x = 1;\nconsole.log(x);")

    def test_prose_not_code(self):
        assert not is_code_heuristic(
            "The import duties on goods were raised last year, a decision that "
            "affected trade across the region in significant ways.")


class TestProfanity:
    def test_hit(self):
        assert contains_profanity("what the fuck is this")

    def test_word_boundary(self):
        # 'Scunthorpe problem': profanity must match whole words only
        assert not contains_profanity("the classic assessment of Scunthorpe")

    def test_clean(self):
        assert not contains_profanity("a perfectly polite sentence")


MULTILINGUAL = {
    'zh': "这是一个完全正常的中文句子，用于测试默认的质量过滤器是否会错误地拒绝非英语数据。这里有足够的字符和内容。",
    'ja': "これは日本語の文章で、品質フィルターが正しく動作するかどうかを確認するためのものです。よろしくお願いします。",
    'ru': "Привет, это совершенно нормальное предложение на русском языке для проверки фильтров качества.",
    'ar': "هذه جملة عربية عادية تماما تستخدم لاختبار ما إذا كانت مرشحات الجودة تعمل بشكل صحيح.",
    'th': "นี่คือประโยคภาษาไทยปกติที่ใช้ทดสอบว่าตัวกรองคุณภาพทำงานได้อย่างถูกต้องหรือไม่",
}


class TestMultilingualDefaults:
    """Default gates must not discard text just because it isn't English."""

    def _cli_defaults(self):
        from sanitizer_pro.cli import build_parser
        return build_parser().parse_args(['--input', 'x', '--output', 'y'])

    def test_default_gates_keep_non_english(self):
        args = self._cli_defaults()
        for lang, text in MULTILINGUAL.items():
            assert _check_quality_reason(text, args) is None, lang

    def test_api_defaults_match_cli(self):
        from sanitizer_pro import SanitizerConfig
        assert SanitizerConfig().min_ascii_ratio == self._cli_defaults().min_ascii_ratio == 0.0

    def test_ascii_gate_still_available(self):
        args = self._cli_defaults()
        args.min_ascii_ratio = 0.85
        assert 'ASCII' in _check_quality_reason(MULTILINGUAL['ru'], args)

    def test_unspaced_scripts_count_per_character(self):
        from sanitizer_pro.quality import words_of
        assert len(words_of("中文句子")) == 4
        assert words_of("GPT-4 和 Claude 很强") == ['GPT', '4', '和', 'Claude', '很', '强']
        assert words_of("plain english words") == ['plain', 'english', 'words']

    def test_cjk_too_few_words_still_rejected(self):
        args = self._cli_defaults()
        args.min_chars = 1
        assert 'too few words' in _check_quality_reason("你好", args)
