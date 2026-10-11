"""
Leave Planner Agent
--------------------
Feeds a structured holiday calendar to tools (not raw prompt text) and lets
a Strands agent answer natural-language leave-planning
questions by calling those tools. Date math is done in Python, so the
model never has to "calculate" days itself -- it only orchestrates.

Run:
    pip install strands-agents --break-system-packages
    python3 leave_planner_agent.py
"""

import json
from datetime import date, timedelta
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# ---------------------------------------------------------------------------
# 1. Holiday data -- this is the thing you "feed" to the model. Keep it as
#    structured data on disk / in a DB, NOT pasted into the system prompt.
#    Swap this out for a JSON file, a DB query, or an API call later.
# ---------------------------------------------------------------------------
HOLIDAYS = [
    {"date": "2026-09-14", "name": "Ganesh Chaturthi"},
    {"date": "2026-09-25", "name": "Ananth Chaturdashi"},
    {"date": "2026-10-02", "name": "Gandhi Jayanti"},
    {"date": "2026-10-20", "name": "Dasara"},
    {"date": "2026-11-08", "name": "Diwali (Laxmi Pujan)"},
    {"date": "2026-11-10", "name": "Diwali (Bali Pratipada)"},
    {"date": "2026-11-11", "name": "Diwali (Bhaubeej)"},
    {"date": "2026-12-25", "name": "Christmas"},
]

_holiday_map = {h["date"]: h["name"] for h in HOLIDAYS}


def _parse(d: str) -> date:
    return date.fromisoformat(d)


# ---------------------------------------------------------------------------
# 2. Tools -- the model calls these to get ground-truth answers instead of
#    guessing. Each tool does exactly one job and returns plain data.
# ---------------------------------------------------------------------------

@tool
def is_holiday(iso_date: str) -> str:
    """Check whether a specific date (YYYY-MM-DD) is a public/company holiday.

    Args:
        iso_date: Date to check, formatted YYYY-MM-DD.
    """
    name = _holiday_map.get(iso_date)
    if name:
        return json.dumps({"date": iso_date, "is_holiday": True, "name": name})
    return json.dumps({"date": iso_date, "is_holiday": False})


@tool
def list_holidays_in_range(start_date: str, end_date: str) -> str:
    """List all holidays between two dates, inclusive (YYYY-MM-DD each).

    Args:
        start_date: Range start, YYYY-MM-DD.
        end_date: Range end, YYYY-MM-DD.
    """
    s, e = _parse(start_date), _parse(end_date)
    matches = [h for h in HOLIDAYS if s <= _parse(h["date"]) <= e]
    return json.dumps(matches)


@tool
def next_holiday_after(iso_date: str) -> str:
    """Find the next upcoming holiday strictly after the given date.

    Args:
        iso_date: Reference date, YYYY-MM-DD.
    """
    ref = _parse(iso_date)
    upcoming = sorted(
        (h for h in HOLIDAYS if _parse(h["date"]) > ref),
        key=lambda h: h["date"],
    )
    return json.dumps(upcoming[0]) if upcoming else json.dumps(None)


@tool
def plan_leave_bridge(leave_start: str, leave_end: str) -> str:
    """Given a proposed leave range, return the holidays inside it, the
    weekends inside it, and the total number of ACTUAL leave days you'd
    need to apply for (excluding weekends and holidays that fall inside
    the range).

    Args:
        leave_start: First day of proposed leave, YYYY-MM-DD.
        leave_end: Last day of proposed leave, YYYY-MM-DD.
    """
    s, e = _parse(leave_start), _parse(leave_end)
    if e < s:
        return json.dumps({"error": "leave_end is before leave_start"})

    total_days = (e - s).days + 1
    cursor = s
    weekends, holidays_hit, leave_days_needed = [], [], 0

    while cursor <= e:
        iso = cursor.isoformat()
        if cursor.weekday() >= 5:  # Sat/Sun
            weekends.append(iso)
        elif iso in _holiday_map:
            holidays_hit.append({"date": iso, "name": _holiday_map[iso]})
        else:
            leave_days_needed += 1
        cursor += timedelta(days=1)

    return json.dumps({
        "range": f"{leave_start} to {leave_end}",
        "total_calendar_days": total_days,
        "weekends_in_range": weekends,
        "holidays_in_range": holidays_hit,
        "actual_leave_days_needed": leave_days_needed,
    })


# ---------------------------------------------------------------------------
# 3. Agent -- Ollam model + the tools above. The system prompt tells it
#    HOW to behave, not WHAT the holidays are -- that stays in the tools.
# ---------------------------------------------------------------------------


# Local with Ollama (no cloud APIs needed)
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="gemma4:latest",
)


agent = Agent(
    model=ollama_model,

    tools=[is_holiday, list_holidays_in_range, next_holiday_after, plan_leave_bridge],
    system_prompt=(
        "You are a leave-planning assistant. You have tools that hold the "
        "authoritative holiday calendar and can do date math. "
        "Always call a tool to check dates or count days -- never compute "
        "or guess date math yourself. Give short, direct answers, and "
        "mention specific dates/day counts from the tool results."
    ),
)


if __name__ == "__main__":
    print("Leave Planner Agent -- ask a question, or type 'exit' to quit.")
    print("e.g. \"Is 14th Sep 2026 a holiday?\" or \"How many leave days if I take 21-27 Sep 2026?\"\n")
 
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
 
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
 
        answer = agent(question)
        print(f"Agent: {answer}\n")