"""
Steady Step - booking logic
===========================
Bookings are kept in the app session (st.session_state) - SIMULATED storage.
A real deployment would save them in the University's booking system.
"""

from datetime import date

ADVISERS = {
    "Mia Chen": "Academic Success Adviser - study skills, catching up on work",
    "Jordan Williams": "Wellbeing & Support Adviser - stress, balance, personal challenges",
    "Ari Patel": "Course Planning Adviser - workload, changing papers, degree planning",
}
TIME_SLOTS = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def make_booking(bookings, learner_id, adviser, day, time, note=""):
    """Validate and add a booking. Returns (ok, message).

    Rules:
      - adviser must exist and the time must be a listed slot
      - date cannot be in the past
      - an adviser cannot be double-booked for the same date and time
    """
    if adviser not in ADVISERS:
        return False, f"Unknown adviser: {adviser}"
    if time not in TIME_SLOTS:
        return False, f"{time} is not an available time slot"
    if day < date.today():
        return False, "Please choose today or a future date"
    for b in bookings:
        if b["adviser"] == adviser and b["date"] == day and b["time"] == time:
            return False, f"{adviser} is already booked at {time} on {day:%a %d %b}. Please pick another time."

    bookings.append({"learner_id": learner_id, "adviser": adviser, "date": day,
                     "time": time, "note": note.strip(), "status": "Confirmed"})
    return True, f"Booked with {adviser} on {day:%A %d %B} at {time}."
