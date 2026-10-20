"""Run with:  python -m pytest -v"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rules  # noqa: E402
from scenarios import ALL_SCENARIOS  # noqa: E402


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.__name__)
def test_scenario(scenario):
    result = scenario()
    assert result["passed"], f"{result['id']}: expected {result['expected']}, got {result['actual']}"


def test_protected_attributes_not_used_in_flag():
    """The flag must be identical if every protected attribute is scrambled."""
    data = rules.load_data()
    before = rules.apply_rules(rules.build_checkpoint_features(data)).flagged
    data["learners"][rules.PROTECTED_COLUMNS] = data["learners"][rules.PROTECTED_COLUMNS].sample(
        frac=1, random_state=1).to_numpy()
    after = rules.apply_rules(rules.build_checkpoint_features(data)).flagged
    assert before.equals(after)


def test_every_flag_has_a_reason():
    data = rules.load_data()
    res = rules.apply_rules(rules.build_checkpoint_features(data))
    assert (res[res.flagged].reasons != "No indicators").all()
