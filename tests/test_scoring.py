"""Tests for heuristic and perplexity quality scoring."""
import pytest

from sanitizer_pro.scoring import (
    HeuristicScorer, PerplexityScorer, _band, make_scorer, perplexity_to_score,
)
from sanitizer_pro.utils import ConfigurationError

GOOD_PROSE = (
    "The committee reviewed the proposal in detail and concluded that the plan "
    "was feasible, although several members raised concerns about the projected "
    "costs and the ambitious timeline for the first phase of construction.")

GIBBERISH = "asdf qwer zxcv hjkl uiop mnbv fdsa rewq vcxz lkjh poiu vbnm qazw sxed"
SPAM = "buy now click here buy now click here buy now click here buy now click here"
SYMBOL_SOUP = "@#$% ^&*( )!@# $%^& *()! @#$% ^&*( )!@# $%^&"


class TestBand:
    def test_inside_plateau(self):
        assert _band(0.5, 0.0, 0.2, 0.8, 1.0) == 1.0

    def test_outside(self):
        assert _band(-0.1, 0.0, 0.2, 0.8, 1.0) == 0.0
        assert _band(1.5, 0.0, 0.2, 0.8, 1.0) == 0.0

    def test_ramps(self):
        assert 0.0 < _band(0.1, 0.0, 0.2, 0.8, 1.0) < 1.0
        assert 0.0 < _band(0.9, 0.0, 0.2, 0.8, 1.0) < 1.0


class TestHeuristicScorer:
    def setup_method(self):
        self.s = HeuristicScorer()

    def test_good_prose_scores_high(self):
        assert self.s.score(GOOD_PROSE) >= 0.7

    def test_gibberish_scores_below_prose(self):
        assert self.s.score(GIBBERISH) < self.s.score(GOOD_PROSE)

    def test_repetitive_spam_scores_low(self):
        assert self.s.score(SPAM) < 0.5

    def test_symbol_soup_scores_low(self):
        assert self.s.score(SYMBOL_SOUP) < 0.3

    def test_empty_and_tiny(self):
        assert self.s.score("") == 0.0
        assert self.s.score("one two") == 0.0

    def test_score_in_range(self):
        for text in (GOOD_PROSE, GIBBERISH, SPAM, SYMBOL_SOUP, "x " * 500):
            assert 0.0 <= self.s.score(text) <= 1.0


class TestPerplexityMapping:
    def test_endpoints(self):
        assert perplexity_to_score(5) == 1.0
        assert perplexity_to_score(10) == 1.0
        assert perplexity_to_score(10_000) == 0.0
        assert perplexity_to_score(1e9) == 0.0

    def test_monotone(self):
        scores = [perplexity_to_score(p) for p in (10, 50, 200, 1000, 5000, 10_000)]
        assert scores == sorted(scores, reverse=True)

    def test_midpoint(self):
        assert 0.4 < perplexity_to_score(316) < 0.6  # sqrt(10*10000) ≈ geometric middle


class TestPerplexityScorer:
    def test_injected_ppl_fn(self):
        s = PerplexityScorer(_ppl_fn=lambda text: 10.0)
        assert s.score(GOOD_PROSE) == 1.0
        s = PerplexityScorer(_ppl_fn=lambda text: 10_000.0)
        assert s.score(GOOD_PROSE) == 0.0

    def test_empty_text_zero(self):
        s = PerplexityScorer(_ppl_fn=lambda text: 10.0)
        assert s.score("   ") == 0.0


class TestMakeScorer:
    def test_heuristic(self):
        assert make_scorer('heuristic').backend_name == 'heuristic'

    def test_unknown(self):
        with pytest.raises(ConfigurationError):
            make_scorer('vibes')
