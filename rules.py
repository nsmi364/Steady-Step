"""
Steady Step - core logic (data -> indicators -> flag -> reason)
===============================================================
This file is the "processing / decision logic" box in our architecture diagram.

Checkpoint: end of Week 3 (day 21), just after the first quiz (due day 14).
Only data available ON OR BEFORE day 21 is used. Final outcomes, withdrawal
and protected attributes are NEVER inputs to the flag.

Three indicators (same as Report 2, Appendix H):
  1. Low LMS activity   - total interactions days -7..21 in the bottom X% of the course
  2. First quiz issue   - first assessment late, or not submitted by day 21
  3. Late registration  - registered after day 0 (course start)

Rule: a learner is flagged when at least N indicators are true (default N = 2).
A flag is only a SUGGESTION for an adviser to check in. The adviser decides.
"""

from pathlib import Path

import pandas as pd

CHECKPOINT_DAY = 21
DATA_DIR = Path(__file__).parent / "data"

# Default rule settings (the adviser can change these with sliders in the app)
DEFAULT_ACTIVITY_PERCENTILE = 25   # bottom 25% of the course = low activity
DEFAULT_LATE_REG_DAY = 0           # registering after day 0 = late
DEFAULT_MIN_INDICATORS = 2         # flag when 2 or more indicators are true

PROTECTED_COLUMNS = ["gender_group", "age_group", "area_deprivation_band", "accessibility_support_flag"]

# Plain-language reasons (adviser wording / student wording)
REASONS = {
    "low_activity": (
        "Low Canvas activity in Weeks 1-3 (bottom {pct}% of the course)",
        "You've been on Canvas less than most of your classmates in the first three weeks.",
    ),
    "first_quiz_issue": (
        "First quiz {status}",
        "{student_status} - it's not too late to catch up.",
    ),
    "late_registration": (
        "Registered late (day {reg_day})",
        "You joined the course after it started, so you may have missed some early material.",
    ),
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Read the 7 synthetic CSV tables into a dict of DataFrames."""
    names = ["courses", "assessments", "learners", "registrations",
             "learning_resources", "learning_activity", "submissions"]
    return {n: pd.read_csv(Path(data_dir) / f"{n}.csv") for n in names}


# ---------------------------------------------------------------------------
# Step 1: transform raw tables into one row per learner at the checkpoint
# ---------------------------------------------------------------------------
def build_checkpoint_features(data, checkpoint_day=CHECKPOINT_DAY):
    learners = data["learners"][["learner_id", "course_code"]].copy()

    # (a) LMS activity up to the checkpoint only
    act = data["learning_activity"]
    act = act[act.activity_day <= checkpoint_day]
    clicks = act.groupby("learner_id").interaction_count.sum()
    days = act.groupby("learner_id").activity_day.nunique()
    learners["interactions_to_day21"] = learners.learner_id.map(clicks).fillna(0).astype(int)
    learners["active_days_to_day21"] = learners.learner_id.map(days).fillna(0).astype(int)
    # percentile rank within the learner's own course (0 = least active)
    learners["activity_percentile"] = (
        learners.groupby("course_code").interactions_to_day21.rank(pct=True, method="max") * 100
    ).round(1)

    # (b) first assessment that is due on or before the checkpoint
    a = data["assessments"]
    first = (a[a.assessment_due_day <= checkpoint_day]
             .sort_values("assessment_due_day")
             .groupby("course_code").first()
             .reset_index()[["course_code", "assessment_id", "assessment_due_day"]])
    learners = learners.merge(first, on="course_code", how="left")

    subs = data["submissions"]
    subs = subs[subs.submission_day <= checkpoint_day]          # ignore anything after day 21
    learners = learners.merge(subs[["assessment_id", "learner_id", "submission_day"]],
                              on=["assessment_id", "learner_id"], how="left")
    learners["first_quiz_status"] = "on time"
    learners.loc[learners.submission_day > learners.assessment_due_day, "first_quiz_status"] = "late"
    learners.loc[learners.submission_day.isna(), "first_quiz_status"] = "not submitted"

    # (c) registration day
    reg = data["registrations"].set_index("learner_id").registration_day
    learners["registration_day"] = learners.learner_id.map(reg)

    return learners[["learner_id", "course_code", "interactions_to_day21", "active_days_to_day21",
                     "activity_percentile", "first_quiz_status", "registration_day"]]


# ---------------------------------------------------------------------------
# Step 2: apply the decision rule and write a reason for every flag
# ---------------------------------------------------------------------------
def apply_rules(features,
                activity_percentile=DEFAULT_ACTIVITY_PERCENTILE,
                late_reg_day=DEFAULT_LATE_REG_DAY,
                min_indicators=DEFAULT_MIN_INDICATORS):
    df = features.copy()
    df["low_activity"] = df.activity_percentile <= activity_percentile
    df["first_quiz_issue"] = df.first_quiz_status.isin(["late", "not submitted"])
    df["late_registration"] = df.registration_day > late_reg_day

    indicators = ["low_activity", "first_quiz_issue", "late_registration"]
    df["indicator_count"] = df[indicators].sum(axis=1)
    df["flagged"] = df.indicator_count >= min_indicators

    def reasons(row, who):
        i = 0 if who == "adviser" else 1
        out = []
        if row.low_activity:
            out.append(REASONS["low_activity"][i].format(pct=activity_percentile))
        if row.first_quiz_issue:
            student_status = ("Your first quiz came in late" if row.first_quiz_status == "late"
                              else "We haven't seen your first quiz yet")
            out.append(REASONS["first_quiz_issue"][i].format(status=row.first_quiz_status,
                                                             student_status=student_status))
        if row.late_registration:
            out.append(REASONS["late_registration"][i].format(reg_day=int(row.registration_day)))
        return out

    df["reasons"] = df.apply(lambda r: "; ".join(reasons(r, "adviser")) or "No indicators", axis=1)
    df["student_reasons"] = df.apply(lambda r: reasons(r, "student"), axis=1)
    df["status"] = df.apply(status_label, axis=1)
    return df


def status_label(row):
    """Supportive, non-labelling status text (Chang, Chen & Lee, 2025)."""
    if row.flagged:
        return "Suggested check-in"
    if row.indicator_count >= 1:
        return "Worth keeping an eye on"
    return "On track"


# ---------------------------------------------------------------------------
# Step 3: fairness check (protected attributes used ONLY here)
# ---------------------------------------------------------------------------
def fairness_table(results, learners, gap_threshold=10, min_group_size=30):
    df = results[["learner_id", "flagged"]].merge(learners, on="learner_id")
    overall = df.flagged.mean() * 100
    rows = []
    for col in PROTECTED_COLUMNS:
        for group, g in df.groupby(col):
            rate = g.flagged.mean() * 100
            gap = rate - overall
            rows.append({
                "attribute": col, "group": group, "n": len(g),
                "flag_rate_%": round(rate, 1), "gap_vs_overall_pts": round(gap, 1),
                "needs_review": abs(gap) > gap_threshold,
                "small_group": len(g) < min_group_size,
            })
    return pd.DataFrame(rows), round(overall, 1)


# ---------------------------------------------------------------------------
# Step 4: validation against final outcome (outcome used ONLY for checking)
# ---------------------------------------------------------------------------
def validation_summary(results, learners):
    df = results[["learner_id", "flagged"]].merge(learners[["learner_id", "final_course_outcome"]])
    df["struggled"] = df.final_course_outcome.isin(["Fail", "Withdrawn"])
    tp = int((df.flagged & df.struggled).sum())
    fp = int((df.flagged & ~df.struggled).sum())
    fn = int((~df.flagged & df.struggled).sum())
    tn = int((~df.flagged & ~df.struggled).sum())
    return {
        "flagged": tp + fp, "struggled": tp + fn,
        "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
        "precision_%": round(100 * tp / (tp + fp), 1) if tp + fp else 0.0,
        "recall_%": round(100 * tp / (tp + fn), 1) if tp + fn else 0.0,
    }
