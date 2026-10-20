"""
Steady Step - early-support dashboard (Streamlit)
=================================================
INFOMGMT 399 Group 2 (The Big 4) - Proposal 2: Student Dropout Risk

Run:  streamlit run app.py

Architecture (matches our slide diagram):
  data/*.csv  ->  rules.build_checkpoint_features()  ->  rules.apply_rules()
              ->  Adviser view / Student view  ->  booking.make_booking()  ->  Calendar
  Protected attributes -> rules.fairness_table() only (never the flag)
"""

import inspect
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import rules
from booking import ADVISERS, TIME_SLOTS, make_booking
from scenarios import run_all

st.set_page_config(page_title="Steady Step", page_icon="🌱", layout="wide")


# ---------------------------------------------------------------------------
# Data (cached so it loads once)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    data = rules.load_data()
    return data, rules.build_checkpoint_features(data)


data, features = get_data()

# Session state = simulated storage for this demo session
st.session_state.setdefault("bookings", [])
st.session_state.setdefault("adviser_actions", {})

# ---------------------------------------------------------------------------
# Sidebar: sign-in (test accounts only) + navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🌱 Steady Step")
st.sidebar.caption("Early-support dashboard · synthetic data only · checkpoint: end of Week 3 (day 21)")

role = st.sidebar.radio("Signed in as (test account)", ["Adviser", "Student"], horizontal=True)

if role == "Adviser":
    st.sidebar.markdown("**Account:** `adviser.demo`")
    page = st.sidebar.radio("Go to", [
        "Learners to check in with", "Bookings calendar", "Fairness check",
        "Validation", "Data & logic", "Test scenarios", "About & status",
    ])
    with st.sidebar.expander("⚙️ Rule settings", expanded=False):
        pct = st.slider("Low activity = bottom X% of course", 10, 50, rules.DEFAULT_ACTIVITY_PERCENTILE, 5)
        late_day = st.slider("Late registration = registered after day", -7, 7, rules.DEFAULT_LATE_REG_DAY)
        min_ind = st.slider("Flag when at least N indicators", 1, 3, rules.DEFAULT_MIN_INDICATORS)
else:
    learner_id = st.sidebar.selectbox("Account (test learner)", features.learner_id, index=0)
    page = st.sidebar.radio("Go to", ["My progress", "Book a conversation", "My calendar"])
    pct, late_day, min_ind = (rules.DEFAULT_ACTIVITY_PERCENTILE, rules.DEFAULT_LATE_REG_DAY,
                              rules.DEFAULT_MIN_INDICATORS)

# The core processing step - re-runs every time a setting changes
results = rules.apply_rules(features, pct, late_day, min_ind)

STATUS_COLOUR = {"On track": "🟢", "Worth keeping an eye on": "🟡", "Suggested check-in": "🟠"}


def booking_form(default_learner=None, key="bk"):
    """Shared booking form (student page and adviser page)."""
    cols = st.columns(len(ADVISERS))
    for col, (name, desc) in zip(cols, ADVISERS.items()):
        col.markdown(f"**{name}**  \n{desc}")
    with st.form(key):
        who = default_learner or st.selectbox("Learner", results[results.flagged].learner_id)
        adviser = st.selectbox("Adviser", list(ADVISERS))
        c1, c2 = st.columns(2)
        day = c1.date_input("Date", value=date.today() + timedelta(days=1))
        time = c2.selectbox("Time", TIME_SLOTS)
        note = st.text_area("What would you like to talk about? (optional)", max_chars=300)
        if st.form_submit_button("Book conversation", type="primary"):
            ok, msg = make_booking(st.session_state.bookings, who, adviser, day, time, note)
            if ok:
                st.success(f"✅ {msg}")
                st.info("📧 Simulated email: a confirmation would be sent to the learner's university email.")
            else:
                st.error(msg)


def show_calendar(bookings):
    if not bookings:
        st.info("No conversations booked yet.")
        return
    df = pd.DataFrame(bookings).sort_values(["date", "time"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%a %d %b %Y")
    st.dataframe(df, hide_index=True, width="stretch")


# ===========================================================================
# STUDENT PAGES
# ===========================================================================
if role == "Student":
    me = results[results.learner_id == learner_id].iloc[0]   # a student only ever sees their own row

    if page == "My progress":
        st.title("My progress")
        st.caption(f"{learner_id} · {me.course_code} · based on your first three weeks")

        if st.session_state.get("show_book_cta", True) and me.indicator_count >= 1:
            st.markdown("#### 💬 Want a quick chat with an adviser? It's free and confidential.")
            st.caption("Use **Book a conversation** in the sidebar.")

        st.subheader(f"{STATUS_COLOUR[me.status]} {me.status}")
        if me.student_reasons:
            st.write("Here's what we noticed:")
            for r in me.student_reasons:
                st.markdown(f"- {r}")
            st.write("Lots of students go through this. A short conversation can help you get back on track.")
        else:
            st.write("Nice work - you're engaging well so far. Keep it up! 🎉")

        c1, c2, c3 = st.columns(3)
        c1.metric("Canvas interactions (days -7 to 21)", int(me.interactions_to_day21))
        c2.metric("Active days", int(me.active_days_to_day21))
        c3.metric("First quiz", me.first_quiz_status.capitalize())

        act = data["learning_activity"]
        mine = act[(act.learner_id == learner_id) & (act.activity_day <= rules.CHECKPOINT_DAY)]
        if len(mine):
            daily = mine.groupby("activity_day").interaction_count.sum().reset_index()
            st.altair_chart(alt.Chart(daily).mark_bar().encode(
                x=alt.X("activity_day:Q", title="Course day"),
                y=alt.Y("interaction_count:Q", title="Canvas interactions"),
            ).properties(height=220), width="stretch")
        st.caption("Only you and (with your consent) your adviser can see this page.")

    elif page == "Book a conversation":
        st.title("Book a conversation")
        st.write("Choose who you'd like to talk to. You decide whether to book - nothing happens automatically.")
        booking_form(default_learner=learner_id, key="student_booking")

    elif page == "My calendar":
        st.title("My calendar")
        show_calendar([b for b in st.session_state.bookings if b["learner_id"] == learner_id])

# ===========================================================================
# ADVISER PAGES
# ===========================================================================
else:
    if page == "Learners to check in with":
        st.title("Learners to check in with")
        st.caption("Suggestions only - you decide what happens next. Protected attributes are not shown or used.")

        course = st.selectbox("Course", ["All"] + sorted(results.course_code.unique()))
        view = results if course == "All" else results[results.course_code == course]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Learners", len(view))
        c2.metric("Suggested check-in", int(view.flagged.sum()))
        c3.metric("Worth keeping an eye on", int((view.status == "Worth keeping an eye on").sum()))
        c4.metric("Flag rate", f"{100 * view.flagged.mean():.1f}%")

        flagged = view[view.flagged].copy()
        flagged["adviser_decision"] = flagged.learner_id.map(
            st.session_state.adviser_actions).fillna("Not reviewed")
        edited = st.data_editor(
            flagged[["learner_id", "course_code", "indicator_count", "reasons", "adviser_decision"]],
            column_config={"adviser_decision": st.column_config.SelectboxColumn(
                "Your decision", options=["Not reviewed", "Contact learner", "Monitor", "No action needed"])},
            disabled=["learner_id", "course_code", "indicator_count", "reasons"],
            hide_index=True, width="stretch", key="decisions",
        )
        st.session_state.adviser_actions.update(dict(zip(edited.learner_id, edited.adviser_decision)))

        st.subheader("Learner detail")
        pick = st.selectbox("Choose a learner", flagged.learner_id if len(flagged) else results.learner_id)
        row = results[results.learner_id == pick].iloc[0]
        st.markdown(f"**{pick}** - {STATUS_COLOUR[row.status]} {row.status}")
        st.table(pd.DataFrame({
            "Indicator": ["Low Canvas activity", "First quiz late/missing", "Late registration"],
            "Value at day 21": [f"{row.interactions_to_day21} interactions (percentile {row.activity_percentile})",
                                row.first_quiz_status, f"registered day {int(row.registration_day)}"],
            "Triggered": ["Yes" if row.low_activity else "No",
                          "Yes" if row.first_quiz_issue else "No",
                          "Yes" if row.late_registration else "No"],
        }))
        with st.expander("Book a conversation for this learner"):
            booking_form(default_learner=pick, key="adviser_booking")

    elif page == "Bookings calendar":
        st.title("Bookings calendar")
        show_calendar(st.session_state.bookings)

    elif page == "Fairness check":
        st.title("Fairness check")
        st.caption("Protected attributes are used ONLY here, to audit the flag - never to create it.")
        gap = st.slider("Review if a group's flag rate differs from overall by more than (points)", 5, 20, 10)
        table, overall = rules.fairness_table(results, data["learners"], gap_threshold=gap)
        st.metric("Overall flag rate", f"{overall}%")

        attr = st.selectbox("Attribute", rules.PROTECTED_COLUMNS)
        t = table[table.attribute == attr]
        bars = alt.Chart(t).mark_bar().encode(
            y=alt.Y("group:N", title=None),
            x=alt.X("flag_rate_%:Q", title="Flag rate (%)"),
            color=alt.condition("datum.needs_review", alt.value("#d9534f"), alt.value("#5b8def")),
            tooltip=["group", "n", "flag_rate_%", "gap_vs_overall_pts"],
        )
        rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=60 + 40 * len(t)), width="stretch")
        st.dataframe(t.drop(columns="attribute"), hide_index=True, width="stretch")

        flagged_groups = table[table.needs_review]
        if flagged_groups.empty:
            st.success(f"No group differs by more than {gap} points.")
        for _, g in flagged_groups.iterrows():
            note = " (small group - result not reliable)" if g.small_group else ""
            st.warning(f"Review: {g.attribute} = {g.group} is {g.gap_vs_overall_pts:+} points{note}.")
        st.caption("Response plan: if a reliable gap appears, review the indicator thresholds with the "
                   "teaching team before the tool is used, and report the change.")

    elif page == "Validation":
        st.title("Validation against final outcome")
        st.caption("final_course_outcome is used ONLY here, to check the rule. It is never an input.")
        v = rules.validation_summary(results, data["learners"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged at day 21", v["flagged"])
        c2.metric("Precision (flagged who later failed/withdrew)", f"{v['precision_%']}%")
        c3.metric("Recall (failed/withdrew who were flagged)", f"{v['recall_%']}%")
        st.table(pd.DataFrame(
            [[v["true_pos"], v["false_pos"]], [v["false_neg"], v["true_neg"]]],
            index=["Flagged", "Not flagged"], columns=["Failed/Withdrew", "Passed"]))
        st.info("Try the rule settings in the sidebar: lowering N to 1 catches more learners (recall up) "
                "but adds more false alarms (precision down). Results are on synthetic data only.")

    elif page == "Data & logic":
        st.title("Data & logic")
        st.markdown("**Data source:** 7 synthetic CSV tables we generated (`data/generate_data.py`) "
                    "following the teaching team's data structure.")
        st.dataframe(pd.DataFrame({"table": list(data), "rows": [len(d) for d in data.values()],
                                   "columns": [", ".join(d.columns) for d in data.values()]}),
                     hide_index=True, width="stretch")
        tbl = st.selectbox("Preview a table", list(data))
        st.dataframe(data[tbl].head(20), width="stretch")

        st.subheader("What the flag uses (day 21 only)")
        st.markdown("- **Used:** `learning_activity` (days -7 to 21), first quiz `submissions`, "
                    "`registration_day`\n- **Not used:** final outcome, withdrawal, scores after day 21, "
                    "gender, age, deprivation, accessibility")
        st.subheader("Core rule (rules.apply_rules)")
        src = inspect.getsource(rules.apply_rules).split("def reasons")[0]
        st.code(src, language="python")

    elif page == "Test scenarios":
        st.title("Test scenarios")
        st.caption("These call the real functions in rules.py and booking.py. Same tests run in pytest.")
        tests = run_all()
        tests["result"] = tests.passed.map({True: "✅ Pass", False: "❌ Fail"})
        st.dataframe(tests[["id", "type", "input", "expected", "actual", "result"]],
                     hide_index=True, width="stretch")
        st.metric("Passed", f"{int(tests.passed.sum())} / {len(tests)}")

    elif page == "About & status":
        st.title("About & status")
        st.markdown("""
**Steady Step** helps advisers notice, at the end of Week 3, which learners may benefit from a
check-in - and helps learners book a conversation. The adviser always decides.

| Feature | Status |
|---|---|
| Synthetic dataset (7 tables, 400 learners) | ✅ Working |
| Day-21 indicators, flag rule and reasons | ✅ Working |
| Adviser view and decisions | ✅ Working (decisions kept for this session only) |
| Student view, booking, calendar | ✅ Working (bookings kept for this session only) |
| Fairness check and validation | ✅ Working (synthetic data only) |
| Email confirmation | 🟡 Simulated (on-screen message only) |
| Login | 🟡 Simulated (test accounts, no passwords) |
| Real Canvas / LMS data, consent flow, data retention | ⚪ Planned - needs University approval |

)
