"""
Steady Step - test scenarios
============================
Each scenario has: input, expected result, actual result, pass/fail.
The SAME functions are shown in the app ("Test scenarios" page) and run by
pytest (tests/test_rules.py), so the video shows the real code being tested.
"""

from datetime import date, timedelta

import pandas as pd

from booking import make_booking
from rules import apply_rules, build_checkpoint_features


def one_learner(percentile, quiz, reg_day):
    """Hand-made checkpoint row for one fictional learner."""
    return pd.DataFrame([{
        "learner_id": "TEST01", "course_code": "INF200",
        "interactions_to_day21": 0, "active_days_to_day21": 0,
        "activity_percentile": percentile, "first_quiz_status": quiz, "registration_day": reg_day,
    }])


def t1_normal():
    inp = one_learner(10, "not submitted", -5)
    r = apply_rules(inp).iloc[0]
    return {
        "id": "T1", "type": "Normal case",
        "input": "Activity percentile 10, first quiz not submitted, registered day -5",
        "expected": "Flagged; 2 reasons (low activity + first quiz)",
        "actual": f"Flagged={r.flagged}; {r.indicator_count} reasons: {r.reasons}",
        "passed": bool(r.flagged and r.indicator_count == 2),
    }


def t2_boundary_one_indicator():
    inp = one_learner(10, "on time", -5)
    r = apply_rules(inp).iloc[0]
    return {
        "id": "T2", "type": "Boundary - only 1 indicator",
        "input": "Activity percentile 10, first quiz on time, registered day -5",
        "expected": "NOT flagged; status 'Worth keeping an eye on'",
        "actual": f"Flagged={r.flagged}; status '{r.status}'",
        "passed": bool((not r.flagged) and r.status == "Worth keeping an eye on"),
    }


def t3_boundary_threshold():
    inp = one_learner(25.0, "on time", 3)
    r = apply_rules(inp).iloc[0]
    return {
        "id": "T3", "type": "Boundary - exactly on the 25% threshold",
        "input": "Activity percentile exactly 25.0, quiz on time, registered day 3",
        "expected": "Flagged (25 counts as 'bottom 25%'; late registration also true)",
        "actual": f"Flagged={r.flagged}; reasons: {r.reasons}",
        "passed": bool(r.flagged and r.low_activity and r.late_registration),
    }


def t4_no_data_leakage():
    """Activity and a quiz submission AFTER day 21 must be ignored."""
    data = {
        "learners": pd.DataFrame([{"learner_id": "TEST02", "course_code": "INF200"}]),
        "learning_activity": pd.DataFrame([
            {"learner_id": "TEST02", "activity_day": 5, "interaction_count": 3},
            {"learner_id": "TEST02", "activity_day": 40, "interaction_count": 150},  # after checkpoint
        ]),
        "assessments": pd.DataFrame([{"assessment_id": "A9001", "course_code": "INF200",
                                      "assessment_due_day": 14}]),
        "submissions": pd.DataFrame([{"assessment_id": "A9001", "learner_id": "TEST02",
                                      "submission_day": 30}]),                   # after checkpoint
        "registrations": pd.DataFrame([{"learner_id": "TEST02", "registration_day": -10}]),
    }
    f = build_checkpoint_features(data).iloc[0]
    return {
        "id": "T4", "type": "Error/leakage - data after day 21",
        "input": "3 clicks on day 5 + 150 clicks on day 40; quiz submitted day 30",
        "expected": "Only 3 interactions counted; quiz treated as 'not submitted' at day 21",
        "actual": f"{f.interactions_to_day21} interactions; quiz '{f.first_quiz_status}'",
        "passed": bool(f.interactions_to_day21 == 3 and f.first_quiz_status == "not submitted"),
    }


def t5_double_booking():
    bookings, day = [], date.today() + timedelta(days=2)
    ok1, _ = make_booking(bookings, "L0001", "Mia Chen", day, "10:00")
    ok2, msg2 = make_booking(bookings, "L0002", "Mia Chen", day, "10:00")
    return {
        "id": "T5", "type": "Error - double booking",
        "input": "Two learners book Mia Chen at the same date and 10:00",
        "expected": "First booking confirmed; second rejected with a clear message",
        "actual": f"First ok={ok1}; second ok={ok2}: '{msg2}'",
        "passed": bool(ok1 and not ok2 and len(bookings) == 1),
    }


ALL_SCENARIOS = [t1_normal, t2_boundary_one_indicator, t3_boundary_threshold,
                 t4_no_data_leakage, t5_double_booking]


def run_all():
    return pd.DataFrame([s() for s in ALL_SCENARIOS])
