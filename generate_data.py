"""
Steady Step - synthetic dataset generator
=========================================
INFOMGMT 399, Group 2 (The Big 4), Proposal 2: Student Dropout Risk

Creates 7 CSV tables that follow the teaching team's data structure:

    courses.csv              course_code, course_offering_period, course_duration_days
    assessments.csv          assessment_id, course_code, ..., assessment_due_day, assessment_weight_percent
    learners.csv             learner_id, course_code, ..., gender_group, ..., final_course_outcome
    registrations.csv        learner_id, course_code, ..., registration_day, withdrawal_day
    learning_resources.csv   learning_resource_id, course_code, ..., learning_activity_type, available weeks
    learning_activity.csv    learner_id, course_code, ..., learning_resource_id, activity_day, interaction_count
    submissions.csv          assessment_id, learner_id, submission_day, prior_credit_flag, assessment_score

ALL DATA IS FICTIONAL. No real student is represented.

How the data is generated (so we can explain it in Q&A):
  1. Every learner gets a hidden "engagement" number (standard normal).
     It is NEVER saved to a CSV - it only drives behaviour.
  2. Higher engagement -> registers earlier, logs in more, submits on time,
     scores higher, withdraws less.
  3. A small, documented link between engagement and some profile fields
     (deprivation, previous attempts, heavy credit load) makes the data
     realistic, so our fairness check has something meaningful to test.
  4. final_course_outcome is calculated LAST from the full-semester data.
     It is used only to validate the flag rule, never as an input.

Run:  python data/generate_data.py      (same seed -> same data every time)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 399                      # fixed seed so the dataset is reproducible
N_LEARNERS = 400
OFFERING = "2026S2"
OUT_DIR = Path(__file__).parent

rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def choice(values, probs, size=None):
    return rng.choice(values, p=probs, size=size)


# ---------------------------------------------------------------------------
# 1. Courses
# ---------------------------------------------------------------------------
courses = pd.DataFrame({
    "course_code": ["INF200", "BUS101", "BUSAN101"],
    "course_offering_period": OFFERING,
    "course_duration_days": [84, 84, 98],      # 12- and 14-week courses (allowed 70-280)
})

# ---------------------------------------------------------------------------
# 2. Assessments (first assessment = Quiz due day 14, i.e. before the
#    day-21 checkpoint, so "first quiz late or missing" can be measured)
# ---------------------------------------------------------------------------
ASSESSMENT_PLAN = [  # (category, due_day, weight)
    ("Quiz", 14, 10),
    ("Assignment", 28, 20),
    ("Reflection", 42, 5),
    ("Project", 56, 25),
    ("Exam", None, 40),          # exam due on the last day of the course
]

rows, a_num = [], 1001
for _, c in courses.iterrows():
    for cat, due, weight in ASSESSMENT_PLAN:
        rows.append({
            "assessment_id": f"A{a_num}",
            "course_code": c.course_code,
            "course_offering_period": OFFERING,
            "assessment_category": cat,
            "assessment_due_day": due if due is not None else c.course_duration_days,
            "assessment_weight_percent": weight,
        })
        a_num += 1
assessments = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. Learners (profile + fairness/audit fields)
# ---------------------------------------------------------------------------
learner_ids = [f"L{i:04d}" for i in range(1, N_LEARNERS + 1)]
learners = pd.DataFrame({
    "learner_id": learner_ids,
    "course_code": choice(courses.course_code, [0.40, 0.35, 0.25], N_LEARNERS),
    "course_offering_period": OFFERING,
    "gender_group": choice(["Female", "Male", "Another/Unknown"], [0.50, 0.46, 0.04], N_LEARNERS),
    "learner_region": choice(["North", "South", "Central", "Online/Other"], [0.30, 0.20, 0.35, 0.15], N_LEARNERS),
    "prior_education_level": choice(["Secondary", "Certificate", "Diploma", "Bachelor", "Postgraduate"],
                                    [0.45, 0.10, 0.15, 0.20, 0.10], N_LEARNERS),
    "area_deprivation_band": choice(["Low", "Medium", "High", "Unknown"], [0.35, 0.35, 0.22, 0.08], N_LEARNERS),
    "age_group": choice(["Under 25", "25-34", "35-44", "45+"], [0.65, 0.20, 0.10, 0.05], N_LEARNERS),
    "previous_attempt_count": choice([0, 1, 2, 3, 4], [0.80, 0.13, 0.05, 0.015, 0.005], N_LEARNERS),
    "credit_load": choice([30, 45, 60, 90, 120], [0.10, 0.15, 0.55, 0.12, 0.08], N_LEARNERS),
    "accessibility_support_flag": choice(["No", "Yes", "Unknown"], [0.82, 0.12, 0.06], N_LEARNERS),
})

# Hidden engagement score (NOT exported)
engagement = rng.normal(0, 1, N_LEARNERS)
engagement -= 0.25 * (learners.area_deprivation_band == "High").to_numpy()
engagement -= 0.30 * (learners.previous_attempt_count > 0).to_numpy()
engagement -= 0.20 * (learners.credit_load >= 90).to_numpy()
learners["_eng"] = engagement
duration = learners.course_code.map(courses.set_index("course_code").course_duration_days)

# ---------------------------------------------------------------------------
# 4. Registrations (registration_day -30..14, withdrawal_day blank/21/45/80)
# ---------------------------------------------------------------------------
late_reg = rng.random(N_LEARNERS) < sigmoid(-2.0 - 1.2 * engagement)
registration_day = np.where(late_reg,
                            rng.integers(1, 15, N_LEARNERS),     # registered after the course started
                            rng.integers(-30, 1, N_LEARNERS))    # registered on time

withdraws = rng.random(N_LEARNERS) < sigmoid(-2.6 - 1.3 * engagement)
withdrawal_day = np.where(withdraws, choice([21, 45, 80], [0.3, 0.4, 0.3], N_LEARNERS), np.nan)

registrations = pd.DataFrame({
    "learner_id": learner_ids,
    "course_code": learners.course_code,
    "course_offering_period": OFFERING,
    "registration_day": registration_day,
    "withdrawal_day": pd.array(withdrawal_day, dtype="Int64"),   # blank when not withdrawn
})

# ---------------------------------------------------------------------------
# 5. Learning resources (12 per course)
# ---------------------------------------------------------------------------
RESOURCE_PLAN = [  # (type, from_week, to_week)  None = blank
    ("Homepage", None, None), ("Page", 0, None), ("Forum", 0, None),
    ("Video", 1, 4), ("Resource", 1, None), ("Quiz", 1, 2),
    ("Video", 3, 7), ("URL", 3, None), ("Resource", 5, None),
    ("Video", 7, 11), ("Page", 9, None), ("Resource", 11, 14),
]
rows, r_num = [], 1
for code in courses.course_code:
    for rtype, wk_from, wk_to in RESOURCE_PLAN:
        rows.append({
            "learning_resource_id": f"R{r_num:04d}",
            "course_code": code,
            "course_offering_period": OFFERING,
            "learning_activity_type": rtype,
            "resource_available_from_week": wk_from,
            "resource_available_to_week": wk_to,
        })
        r_num += 1
resources = pd.DataFrame(rows).astype({"resource_available_from_week": "Int64",
                                       "resource_available_to_week": "Int64"})


def week_of(day):
    """Day 1-7 = week 1, ...; days <= 0 are orientation week 0."""
    return 0 if day <= 0 else (day - 1) // 7 + 1


def available(res, day):
    wk = week_of(day)
    ok_from = res.resource_available_from_week.isna() | (res.resource_available_from_week <= wk)
    ok_to = res.resource_available_to_week.isna() | (res.resource_available_to_week >= wk)
    return res[ok_from & ok_to]


# ---------------------------------------------------------------------------
# 6. Learning activity (one row = learner x resource x day with interactions)
# ---------------------------------------------------------------------------
rows = []
for i, lr in learners.iterrows():
    e = engagement[i]
    res = resources[resources.course_code == lr.course_code]
    start = max(-7, registration_day[i])
    end = int(duration[i]) if np.isnan(withdrawal_day[i]) else int(withdrawal_day[i])
    for day in range(start, end + 1):
        # low-engagement learners also fade over the semester
        p_active = sigmoid(-0.5 + 1.0 * e - (0.8 - 0.4 * e) * day / duration[i])
        if rng.random() > p_active:
            continue
        avail = available(res, day)
        n_res = min(len(avail), rng.integers(1, 4))
        for rid in rng.choice(avail.learning_resource_id, size=n_res, replace=False):
            rows.append({
                "learner_id": lr.learner_id,
                "course_code": lr.course_code,
                "course_offering_period": OFFERING,
                "learning_resource_id": rid,
                "activity_day": day,
                "interaction_count": int(min(200, 1 + rng.poisson(3 * np.exp(0.3 * e)))),
            })
activity = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 7. Submissions
# ---------------------------------------------------------------------------
rows = []
for i, lr in learners.iterrows():
    e = engagement[i]
    for _, a in assessments[assessments.course_code == lr.course_code].iterrows():
        due = a.assessment_due_day
        if not np.isnan(withdrawal_day[i]) and withdrawal_day[i] < due:
            continue                                   # withdrew before it was due
        if rng.random() > sigmoid(2.6 + 1.2 * e):
            continue                                   # never submitted -> no row
        if rng.random() < 0.02:                        # credit from earlier study
            rows.append({"assessment_id": a.assessment_id, "learner_id": lr.learner_id,
                         "submission_day": due, "prior_credit_flag": "Yes", "assessment_score": np.nan})
            continue
        late = rng.random() < sigmoid(-1.8 - 1.0 * e)
        sub_day = due + int(rng.integers(1, 8)) if late else due - int(rng.integers(0, 4))
        score = float(np.clip(rng.normal(70 + 12 * e - (8 if late else 0), 12), 0, 100).round())
        rows.append({"assessment_id": a.assessment_id, "learner_id": lr.learner_id,
                     "submission_day": sub_day, "prior_credit_flag": "No", "assessment_score": score})
submissions = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 8. Final outcome (validation only)
# ---------------------------------------------------------------------------
merged = submissions.merge(assessments[["assessment_id", "assessment_weight_percent"]], on="assessment_id")
merged["score_for_total"] = merged.assessment_score.fillna(70)          # prior credit counts as a pass mark
merged["weighted"] = merged.score_for_total * merged.assessment_weight_percent / 100
total = merged.groupby("learner_id").weighted.sum().reindex(learners.learner_id, fill_value=0).to_numpy()

outcome = np.select([~np.isnan(withdrawal_day), total < 50, total >= 80],
                    ["Withdrawn", "Fail", "High Pass"], default="Pass")
learners["final_course_outcome"] = outcome

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
learners = learners.drop(columns="_eng")
for name, df in {
    "courses": courses, "assessments": assessments, "learners": learners,
    "registrations": registrations, "learning_resources": resources,
    "learning_activity": activity, "submissions": submissions,
}.items():
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"{name:20s} {len(df):>6,} rows")

print("\nOutcome mix:", pd.Series(outcome).value_counts().to_dict())
