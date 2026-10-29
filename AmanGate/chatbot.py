import logging
import os
import re
import threading
from datetime import datetime, timedelta

MODEL_PATH = os.environ.get("CHAT_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "models", "Qwen3.5-2B-Q4_K_M.gguf"))
MODEL_ID = os.path.basename(MODEL_PATH)
CHAT_THREADS = int(os.environ.get("CHAT_THREADS", "0"))   # 0 = let llama.cpp choose

# The help guide. The model only chooses a section; the exact text below is what the admin sees,
# so how-to answers cannot contain invented steps.
GUIDE_SECTIONS = [
    ("What each tab of the admin portal is for (overview of the portal)",
     "The admin portal has four tabs:\n"
     "- Dashboard: today's entries and denied scans, entries per hour, gate status, alerts and recent scans.\n"
     "- Add User: enroll a student, faculty or security member, or a 24-hour visitor.\n"
     "- Database: enrolled users in three groups (Students, Employees, Visitors). Search by name, ID or role, and remove users.\n"
     "- Lock Gates: emergency lockdown, unlocking all gates, and locking or unlocking a single gate.\n"
     "Use the Operations / Analytics switch at the top right to open the System Analytics page."),
    ("Enroll, add or register a new student, faculty member or security user",
     "1. Open the Add User tab.\n"
     "2. Enter the Student/Staff ID (digits only, such as 20231234) and the full name.\n"
     "3. Pick the role (Student, Faculty or Security), then press Male or Female.\n"
     "4. Under Gate Access, Main, Male and Female Gate are ticked; untick any the person may not use, "
     "and tick Vehicle Gate only for people who drive in.\n"
     "5. Press Capture Photos. Three photos are taken after a countdown: look straight, "
     "then turn your head a little to the left, then a little to the right.\n"
     "6. Check the three photos and press Save User, or press Retake."),
    ("Register a visitor or give someone temporary 24-hour access",
     "1. Open the Add User tab.\n"
     "2. Pick Visitor (24 hours) as the role, then enter the Visitor ID (national ID or passport number) and the full name.\n"
     "3. Choose the gender and the gates, then press Capture Photos and Save User.\n"
     "Visitor access expires 24 hours after saving, and the visitor is then removed automatically."),
    ("Update or change a user's photo, name or role",
     "There is no edit button; you enroll the person again with the same ID.\n"
     "1. Open the Add User tab.\n"
     "2. Enter the same ID, with the correct name and role.\n"
     "3. Press Capture Photos, then Save User.\n"
     "4. Confirm when asked to overwrite their face data."),
    ("Remove or delete a user",
     "1. Open the Database tab.\n"
     "2. Choose the group (Students, Employees or Visitors) and find the person with the search box.\n"
     "3. Press Remove on their row.\n"
     "4. Press Remove again in the confirmation dialog. Their face data is deleted permanently."),
    ("Find, search or list enrolled users",
     "Open the Database tab, choose Students, Employees (faculty and security) or Visitors, "
     "and type a name, ID or role in the search box. Use the page buttons at the bottom right to see more users."),
    ("Emergency lockdown of all gates, or unlocking all gates",
     "To lock all gates:\n"
     "1. Open the Lock Gates tab.\n"
     "2. Press Lockdown All, then press Lockdown All again to confirm. Every scan is refused.\n"
     "To unlock all gates, open the Lock Gates tab and press Unlock All. It reopens every closed gate and "
     "lifts the lockdown at once."),
    ("Lock or unlock one specific gate (Main, Male, Female or Vehicle gate)",
     "1. Open the Lock Gates tab.\n"
     "2. Under Individual Gate Control, press Lock Gate or Unlock Gate on that gate (or use its switch). "
     "It takes effect at once, and that gate's terminal refuses scans while it is closed."),
    ("What the dashboard shows, and what the alerts mean",
     "The Dashboard tab shows today's entries and denied scans, granted entries per hour for the last 12 hours, "
     "gate status, active alerts and recent scans. Three or more denied scans within five minutes create a "
     "Multiple failed scans alert. Single alerts are an Unauthorized access attempt (unknown face), "
     "Wrong gate (a known person at a gate they may not use) or Spoof attempt (a screen or printed photo)."),
    ("The System Analytics page: weekly charts and statistics",
     "Use the Operations / Analytics switch at the top right. The Analytics page compares this week with last week: "
     "entries, average match confidence, denied attempts, lockdowns and scan time, with charts of daily entries, "
     "granted vs denied, match confidence, user growth, users by group and a traffic heatmap by hour."),
    ("How the gate terminal and face scanning work, and choosing a terminal's gate",
     "1. Open the Gate Terminal link. Choose which gate this terminal is in the chip at the top right "
     "(Main, Male, Female or Vehicle Gate); it is remembered on that device.\n"
     "2. Press Open Camera. One person at a time looks straight at the camera in good light.\n"
     "3. Blink once when asked; this confirms a real person.\n"
     "4. A match shows Access granted with the name and role. Otherwise it shows Unknown user, "
     "Not permitted at this gate, or Fake face detected for a screen or photo.\n"
     "At the Vehicle Gate, the driver lowers the window and looks at the camera; passengers are ignored."),
    ("The camera is not working, black or showing the wrong camera",
     "1. Allow camera access for this site in the browser settings.\n"
     "2. If the wrong camera is showing, choose the correct one with the camera icon in the address bar.\n"
     "3. Close other apps that may be using the camera.\n"
     "4. Reload the page."),
    ("An enrolled person is not recognised, or an enrollment photo is rejected",
     "Enroll them again on the Add User tab with the same ID: one face only, good light, looking straight for "
     "photo 1 and turning only slightly for photos 2 and 3. If the terminal asks to blink, blink once naturally."),
    ("Admin login, logout, or changing the admin password",
     "Log in at /admin with the admin password and log out with the button at the top right. "
     "The password cannot be changed in the portal: it is set by the ADMIN_PASSWORD environment variable "
     "on the server, and the app must be restarted after changing it."),
    ("Switching between light and dark mode",
     "Use the moon/sun button at the top right."),
]

ROUTER_PROMPT = ("You route questions from the admin of AmanGate, a face recognition gate system, "
                 "to a section of its help guide. Reply with the number of the one section that answers "
                 "the question, and nothing else. Reply 0 if no section answers it or the question is not "
                 "about AmanGate.\n\n" + "\n".join(f"{i}. {title}" for i, (title, _) in enumerate(GUIDE_SECTIONS, 1)))

ACTIONS = r"(lockdown|lock|unlock|remove|delete|enrol\w*|add|register|open|close|change|update|reset|rename|edit)"
# "Can you lock...", "Please delete...", or a bare command such as "Delete user 20231234"
ACTION_REQUEST = (r"^\s*(?:please\s+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?" + ACTIONS + r"\b"
                  r"|^\s*(?:please\s+)?" + ACTIONS + r"\b")

# Clear keywords pick the guide section directly; the model only routes questions that match none of these
GUIDE_KEYWORDS = [
    (r"\bvisitors?\b|\bguests?\b|\btemporary\b|\b24[- ]?hours?\b", 3),
    (r"\bpassword\b|\blog ?(?:in|out)\b|\bsign (?:in|out)\b", 14),
    (r"\blockdown\b|\b(?:all|every) gates?\b", 7),
    (r"\b(?:lock|unlock|close|reopen)\w*\b.*\bgates?\b|\bgates?\b.*\b(?:lock|unlock)", 8),
    (r"\b(?:remove|delete)\b", 5),
    (r"\b(?:update|change|edit|rename|replace)\b.*\b(?:photos?|pictures?|face|name|role|user|details)\b", 4),
    (r"\b(?:enrol\w*|register|add)\b", 2),
    (r"\b(?:search|find|look up|list)\b.*\b(?:users?|students?|people|person|someone|staff)\b", 6),
    (r"\banalytics\b|\bstatistics\b|\bcharts?\b", 10),
    (r"\bcamera\b", 12),
    (r"\bnot (?:being )?recogni[sz]ed\b|\bphoto (?:is |was )?rejected\b", 13),
    (r"\b(?:dark|light) mode\b|\btheme\b", 15),
    (r"\bterminal\b", 11),
    (r"\bdashboard\b|\balerts?\b", 9),
]

_model = _load_error = None
_load_lock, _gen_lock = threading.Lock(), threading.Lock()


def _load_model():
    global _model, _load_error
    with _load_lock:
        if _model is None and _load_error is None:
            try:
                from llama_cpp import Llama
                _model = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=CHAT_THREADS or None, verbose=False)
                logging.info("Chat model %s loaded.", MODEL_ID)
            except Exception as exc:
                _load_error = exc
                logging.exception("Chat model failed to load from %s.", MODEL_PATH)
    return _model is not None


def warm_up_async():
    threading.Thread(target=_load_model, daemon=True).start()


def _still_loading():
    return _model is None and _load_lock.locked()


def model_status():
    return "ready" if _model else "unavailable" if _load_error else "loading"


def _ask_model(question, system, max_new_tokens=220):
    # Qwen chat format; the empty <think> block turns off "thinking" so answers come straight away
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    with _gen_lock:
        started = datetime.now()
        out = _model(prompt, max_tokens=max_new_tokens, temperature=0, stop=["<|im_end|>"])
        seconds = (datetime.now() - started).total_seconds()
    made = out["usage"]["completion_tokens"]
    logging.info("Chat answer: %d tokens in %.1fs (%.1f tokens/s)", made, seconds, made / max(seconds, 0.01))
    text = re.sub(r"<think>.*?</think>", "", out["choices"][0]["text"], flags=re.S)
    return re.sub(r"\*\*|__|`|^#+\s*", "", text, flags=re.M).strip()


def _guide_section(q):
    """The guide section a clear keyword points to, or None if the model has to decide."""
    return next((i for pattern, i in GUIDE_KEYWORDS if re.search(pattern, q.lower())), None)


def _guide_answer(q, index=None):
    if index is None:
        reply = _ask_model(q, ROUTER_PROMPT, max_new_tokens=4)
        number = re.search(r"\d+", reply)
        index = int(number.group()) if number else 0
    if not 1 <= index <= len(GUIDE_SECTIONS):
        return OUT_OF_SCOPE
    title, text = GUIDE_SECTIONS[index - 1]
    if re.search(ACTION_REQUEST, q.lower()):
        return "I can't do that myself, but here is how to do it:\n" + text
    return text


def _who(e):
    return e.get("user_id") or e["name"]


def _tag(e, *details):
    """A person as 'Name (ID 1234, details)'; scans without an ID (unknown faces, old scans) keep just the name."""
    parts = ([f"ID {e['user_id']}"] if e.get("user_id") else []) + [x for x in details if x]
    return f"{e['name']} ({', '.join(parts)})" if parts else e["name"]


def _user_tag(u):
    return f"{u['name']} (ID {u['id']})"


def _n(count, one, many):
    return f"{count} {one if count == 1 else many}"


def _when(dt):
    return f"today at {dt:%H:%M}" if dt.date() == datetime.now().date() else f"on {dt:%d %b %Y} at {dt:%H:%M}"


def _enrolled_at(user):
    try:
        return datetime.fromisoformat(user.get("enrolled_at") or "")
    except ValueError:
        return datetime.min


def _users(q, d):
    if re.search(r"\b(enrol\w*|register\w*|added|joined)\b", q) and (_period(q)[2] != "today" or "today" in q):
        start, end, label = _period(q)
        role = next((r for r in ("student", "faculty", "security", "visitor") if r in q), None)
        users = [u for u in d.get("users", list)() if not role or (u.get("role") or "student") == role]
        hits = [_user_tag(u) for u in sorted(users, key=lambda u: u["name"]) if start <= _enrolled_at(u) <= end]
        listed = f": {_list(hits)}" if 0 < len(hits) <= 8 else ""
        who = {"student": ("student was", "students were"), "faculty": ("faculty member was", "faculty members were"),
               "security": ("security user was", "security users were"), "visitor": ("visitor was", "visitors were")}.get(role, ("user was", "users were"))
        return (f"{_n(len(hits), *who)} enrolled {label}{listed}. "
                "This uses each user's latest enrolment date, so re-enrolling someone updates it.")
    counts = d["user_counts"]()
    total = sum(counts.values())
    role_n = lambda r: _n(counts.get(r, 0), r, r + "s" if r in ("student", "visitor") else r)
    for role in ("student", "faculty", "security", "visitor"):
        if role in q:
            return f"{role_n(role)} enrolled, out of {_n(total, 'user', 'users')}."
    parts = ", ".join(role_n(r) for r in sorted(counts) if counts[r])
    return f"{_n(total, 'user', 'users')} enrolled ({parts})." if total else "No users are enrolled yet."


NAME_QUESTION = r"\b(first name|last name|surname|family name|named|called)\b"


def _users_named(q, d):
    m = re.search(r"\b(first name|last name|surname|family name|named|called)\s+(?:is\s+|of\s+)?([a-z][a-z' -]*?)\s*(?:\?|$|\b(?:who|that|in|enrolled|registered)\b)", q)
    if not m:
        return 'Please give the name, e.g. "How many users have the first name Khaled?".'
    kind, value = m.group(1), m.group(2).strip()
    def matches(name):
        name = name.lower()
        if kind == "first name":
            return name.split()[0] == value
        if kind in ("last name", "surname", "family name"):
            return name == value or name.endswith(" " + value)
        return value in name.split() or name == value
    users = [u for u in d.get("users", list)() if matches(u["name"])]
    hits = [_user_tag(u) for u in sorted(users, key=lambda u: u["name"])]
    which = {"named": "named", "called": "called"}.get(kind, f"with the {kind}")
    listed = f": {_list(hits)}" if 0 < len(hits) <= 8 else ""
    return f"{_n(len(hits), 'enrolled user', 'enrolled users')} {which} {value.title()}{listed}."


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _last(granted):
    def handler(q, d):
        events = sorted((e for e in d["events"]() if e["granted"] == granted), key=lambda e: e["at"], reverse=True)
        if not events:
            return "No access has been granted yet." if granted else "There have been no denied scans yet."
        m = re.search(r"\b(\d+)(?:st|nd|rd|th) most recent\b", q)
        n = int(m.group(1)) if m else 1
        kind = "granted entries" if granted else "denied scans"
        if n > len(events):
            return f"There are only {len(events)} {kind} on record."
        e = events[n - 1]
        if not granted:
            where = "The last denied scan" if n == 1 else f"The {_ordinal(n)} most recent denied scan"
            kind = {"DENIED_GATE": f"{_tag(e)}, at a gate they may not use",
                    "DENIED_SPOOF": "a spoof attempt"}.get(e.get("status"), "an unknown face")
            return f"{where} was {_when(e['at'])}: {kind} ({e['confidence']:.0f}% match)."
        if n == 1:
            reply = f"{_tag(e)} was last granted access {_when(e['at'])} ({e['confidence']:.0f}% match)."
        else:
            reply = (f"The {_ordinal(n)} most recent granted entry was {_tag(e)} "
                     f"{_when(e['at'])} ({e['confidence']:.0f}% match).")
        if all(_who(x) == _who(e) for x in events[:n]):
            other = next((x for x in events[n:] if _who(x) != _who(e)), None)
            if n > 1:
                reply += f" The last {n} entries were all {_tag(e)}."
            if other and n > 1:
                reply += f" Before that, the last other person was {_tag(other)} {_when(other['at'])}."
        return reply
    return handler


NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _period(q):
    """Returns (start, end, label) for the time range named in the question; today by default."""
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ago = re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(day|week|month)s?\s+ago\b", q)
    if ago or "day before yesterday" in q:         # one whole earlier day, week or month
        n, unit = (2, "day") if not ago else (int(ago.group(1)) if ago.group(1).isdigit() else NUMBER_WORDS[ago.group(1)], ago.group(2))
        if unit == "day":
            start = today - timedelta(days=n)
            end, label = start + timedelta(days=1), f"on {start:%a %d %b}"
        elif unit == "week":
            start = today - timedelta(days=today.weekday() + 7 * n)
            end = start + timedelta(days=7)
            label = f"the week of {start:%a %d %b} to {end - timedelta(days=1):%a %d %b}"
        else:
            start = today.replace(day=1)
            for _ in range(n):
                start = (start - timedelta(days=1)).replace(day=1)
            end = (start + timedelta(days=32)).replace(day=1)
            label = f"in {start:%B %Y}"
        return start, end - timedelta(microseconds=1), label
    calendar = re.search(r"(?<!the )\b(?:last|previous)\s+(week|month|year)\b", q)
    if calendar:                                   # "last week" = the previous Monday to Sunday, and so on
        unit = calendar.group(1)
        if unit == "week":
            end = today - timedelta(days=today.weekday())
            start = end - timedelta(days=7)
            label = f"last week ({start:%a %d %b} to {end - timedelta(days=1):%a %d %b})"
        elif unit == "month":
            end = today.replace(day=1)
            start = (end - timedelta(days=1)).replace(day=1)
            label = f"last month ({start:%B %Y})"
        else:
            end = today.replace(month=1, day=1)
            start = end.replace(year=end.year - 1)
            label = f"last year ({start.year})"
        return start, end - timedelta(microseconds=1), label
    m = re.search(r"\b(?:past|last|previous|in|within)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")?\s*(day|week|month|year)s?\b", q)
    if m:
        word = m.group(1) or "1"
        count = int(word) if word.isdigit() else NUMBER_WORDS[word]
        unit = m.group(2)
        label = f"in the past {count} {unit}s" if count != 1 else f"in the past {unit}"
        return now - timedelta(days=count * UNIT_DAYS[unit]), now, label
    month = re.search(r"\b(from|since|in|during)?\s*(" + "|".join(MONTHS) + r")\b", q)
    if month:
        number = MONTHS.index(month.group(2)) + 1
        first = today.replace(year=today.year if number <= today.month else today.year - 1, month=number, day=1)
        name = month.group(2).title()
        if month.group(1) in ("from", "since"):
            return first, now, f"since 1 {name}"
        following = first.replace(year=first.year + 1, month=1) if number == 12 else first.replace(month=number + 1)
        return first, min(following - timedelta(microseconds=1), now), f"in {name}"
    year = re.search(r"\b(from|since|in|during)?\s*(20\d\d)\b", q)
    if year:
        first = today.replace(year=int(year.group(2)), month=1, day=1)
        if year.group(1) in ("from", "since"):
            return first, now, f"since {year.group(2)}"
        return first, min(first.replace(year=first.year + 1) - timedelta(microseconds=1), now), f"in {year.group(2)}"
    if re.search(r"\blast night\b|\bovernight\b", q):
        return today - timedelta(hours=6), today + timedelta(hours=6), "last night (6 pm to 6 am)"
    if "yesterday" in q:
        return today - timedelta(days=1), today, "yesterday"
    if re.search(r"\bthis week\b", q):
        return today - timedelta(days=today.weekday()), now, "this week"
    if re.search(r"\bthis month\b", q):
        return today.replace(day=1), now, "this month"
    if re.search(r"\bthis year\b", q):
        return today.replace(month=1, day=1), now, "this year"
    if re.search(r"\b(all time|ever|in total|so far|overall|total)\b", q):
        return datetime.min, now, "in total"
    return today, now, "today"


def _in_period(events, q):
    start, end, label = _period(q)
    return [e for e in events if start <= e["at"] <= end], label


def _list(items, limit=8):
    items = list(items)
    extra = f" and {len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + extra


def _at(e):
    return f"{e['at']:%a %d %b %H:%M}"


DENIAL_KINDS = [("DENIED", "unknown face", "unknown faces"),
                ("DENIED_GATE", "wrong gate", "wrong gate"),
                ("DENIED_SPOOF", "spoof attempt", "spoof attempts")]


def _count(granted=None):
    def handler(q, d):
        events, label = _in_period(d["events"](), q)
        if granted is None:
            ok = sum(e["granted"] for e in events)
            return f"{_n(len(events), 'scan', 'scans')} {label}: {ok} granted, {len(events) - ok} denied."
        hits = [e for e in events if e["granted"] == granted]
        if granted:
            people = len({_who(e) for e in hits})
            reply = f"{_n(len(hits), 'granted entry', 'granted entries')} {label} ({_n(people, 'person', 'people')})."
            current = {u["id"] for u in d.get("users", list)()}
            unlinked = sorted({_tag(e) for e in hits if e.get("user_id") not in current})
            if current and unlinked:
                reply += (f" {_n(len(unlinked), 'name is', 'names are')} not linked to a current user "
                          f"(older scans, or removed or expired users): {_list(unlinked)}.")
            return reply
        reply = f"{_n(len(hits), 'denied attempt', 'denied attempts')} {label}"
        kinds = [(sum(e.get("status") == k for e in hits), one, many) for k, one, many in DENIAL_KINDS]
        if any(n for n, _, _ in kinds[1:]):
            reply += " (" + ", ".join(_n(n, one, many) for n, one, many in kinds if n) + ")"
        return reply + "."
    return handler


def _time_filter(q):
    """Returns (test, description) for 'after hours' or 'after/before <time>', or None."""
    if re.search(r"after hours|out of hours|at night|overnight", q):
        return (lambda t: t.hour >= 22 or t.hour < 6), "after hours (10 pm to 6 am)"
    m = re.search(r"\b(after|before)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", q)
    if not m:
        if re.search(r"\blast night\b", q):       # the period itself is the time window
            return (lambda t: True), ""
        return None
    hour, minute = int(m.group(2)), int(m.group(3) or 0)
    if minute > 59 or hour > (12 if m.group(4) else 23) or (m.group(4) and hour == 0):
        return None
    if m.group(4):
        hour = hour % 12 + (12 if m.group(4) == "pm" else 0)
    cutoff = hour * 60 + minute
    if m.group(1) == "after":
        return (lambda t: t.hour * 60 + t.minute >= cutoff), f"after {hour:02d}:{minute:02d}"
    return (lambda t: t.hour * 60 + t.minute < cutoff), f"before {hour:02d}:{minute:02d}"


def _by_time(q, d):
    rule = _time_filter(q)
    if not rule:
        return 'I couldn\'t read that time. Try e.g. "Who entered after 10 pm this week?".'
    test, desc = rule
    events, label = _in_period(d["events"](), q)
    hits = [e for e in events if test(e["at"])]
    entries = sorted((e for e in hits if e["granted"]), key=lambda e: e["at"])
    window = f"{desc} {label}".strip()
    reply = (f"{_n(len(entries), 'entry', 'entries')} {window}: "
             + _list(_tag(e, _at(e)) for e in entries) + ".") if entries \
        else f"No entries {window}."
    denied = len(hits) - len(entries)
    return reply + (f" Also {_n(denied, 'denied attempt', 'denied attempts')} in that window." if denied else "")


GRANT_FROM = 65   # app.py grants below a distance of 0.35, i.e. above 65% match
LOW_MATCH = 75    # granted entries below this are close to the threshold


def _low_match(q, d):
    m = re.search(r"\b(?:below|under|less than)\s+(\d{2})\s*%?", q)
    limit = int(m.group(1)) if m else LOW_MATCH
    events, label = _in_period(d["events"](), q)
    hits = sorted((e for e in events if e["granted"] and e["confidence"] < limit), key=lambda e: e["at"])
    if not hits:
        return f"No granted entries below {limit}% match {label}."
    return (f"{_n(len(hits), 'granted entry', 'granted entries')} below {limit}% match {label}: "
            + _list(_tag(e, f"{e['confidence']:.1f}%", _at(e)) for e in hits)
            + f". The system grants from {GRANT_FROM}%, so these were close calls worth reviewing.")


NEAR_MISS = 55    # denied scans from here up to GRANT_FROM are near misses (a look-alike or a photo)

def _summary(q, d):
    """A summary built from fixed sentences, so every name, number and time comes straight from the data."""
    events, label = _in_period(d["events"](), q)
    granted = [e for e in events if e["granted"]]
    denied = [e for e in events if not e["granted"]]
    people = len({_who(e) for e in granted})
    head = (f"{label[0].upper() + label[1:]}: {_n(len(events), 'scan', 'scans')}, {len(granted)} granted and "
            f"{len(denied)} denied ({_n(people, 'person', 'people')} granted).")

    # most serious first
    items = []
    if d["lockdown"]():
        items.append("Emergency lockdown is active.")
    spoofs = sum(e.get("status") == "DENIED_SPOOF" for e in denied)
    if spoofs:
        items.append(f"Spoof attempts (screen or printed photo): {spoofs}.")
    wrong = [e for e in denied if e.get("status") == "DENIED_GATE"]
    if wrong:
        items.append("Wrong gate: " + _list(_tag(e, _at(e)) for e in wrong) + ".")
    bursts, group = 0, []
    for e in denied:
        if group and (e["at"] - group[-1]["at"]).total_seconds() > 300:
            bursts += len(group) >= 3
            group = []
        group.append(e)
    bursts += len(group) >= 3
    if bursts:
        items.append(f"Bursts of 3 or more denied scans within 5 minutes: {bursts}.")
    near = [e for e in denied if e.get("status", "DENIED") == "DENIED" and e["confidence"] >= NEAR_MISS]
    if near:
        items.append(f"Near-miss denied scans ({NEAR_MISS} to {GRANT_FROM}% match, possibly a look-alike or a photo): {len(near)}.")
    late = [e for e in granted if e["at"].hour >= 22 or e["at"].hour < 6]
    if late:
        items.append("After hours (10 pm to 6 am): " + _list(_tag(e, _at(e)) for e in late) + ".")
    weak = [e for e in granted if e["confidence"] < LOW_MATCH]
    if weak:
        items.append(f"Weak match (below {LOW_MATCH}%): " + _list(_tag(e, f"{e['confidence']:.1f}%") for e in weak) + ".")

    if not items:
        return f"{head}\nNothing needs attention; the period was quiet."
    return f"{head}\nNeeds attention:\n" + "\n".join(f"- {x}" for x in items)


def _person(q, d):
    """Find one person by student ID or name; returns None when nobody matches."""
    events = d["events"]()
    people = {u["id"]: u["name"] for u in d.get("users", list)()}
    for e in events:
        if e["granted"]:
            people.setdefault(_who(e), e["name"])         # removed users and older scans without an ID
    ids = set(re.findall(r"\b\d{3,}\b", q)) & set(people)
    if ids:
        found = sorted(ids)
    else:
        words = set(re.findall(r"[a-z]+", q))
        scores = {key: sum(part in words for part in re.findall(r"[a-z]{3,}", name.lower()))
                  for key, name in people.items()}
        best = max(scores.values(), default=0)
        if not best:
            return None
        found = sorted(k for k, v in scores.items() if v == best)
    if len(found) > 1:
        options = ", ".join(f"{people[k]} (ID {k})" if k != people[k] else people[k] for k in found)
        same_name = len({people[k].lower() for k in found}) == 1
        return f"More than one person matches: {options}. Please use " + ("the ID." if same_name else "the full name or the ID.")
    key = found[0]
    entries = [e for e in events if e["granted"] and _who(e) == key]
    label_id = f" (ID {key})" if key != people[key] else ""
    if not entries:
        return f"{people[key]}{label_id} has no recorded entries."
    in_period, label = _in_period(entries, q)
    last = _when(max(e["at"] for e in entries))
    if not in_period:
        return f"{people[key]}{label_id} last entered {last} (no entries {label})."
    return f"{people[key]}{label_id} entered {_n(len(in_period), 'time', 'times')} {label}; last entry {last}."


def _not_in_chat(q, d):
    return ("I don't list inactive users here, because at university scale the answer is a long list "
            "that belongs in an access review report. You can ask about one person instead, "
            "e.g. \"When did Khaled last enter?\".")


def _lockdown(q, d):
    return "Emergency lockdown is ACTIVE; all scans are refused." if d["lockdown"]() \
        else "Lockdown is off; the gates are scanning normally."


LAST_DENIED = r"^(?!.*\bhow many\b)(?=.*\b(last|latest|recent)\b).*\b(denied|rejected|failed)\b"
LAST_GRANTED = (r"^(?!.*\bhow many\b)(?:.*\b(last|latest|recent)\b.*\b(grant\w*|access|enter\w*|entry|admitted|scan)\b"
                r"|.*\bwho\b.*\b(granted|entered|came in)\b)")
BEFORE_THAT = (r"^\s*(?:and\s+)?(?:what about\s+|how about\s+)?(?:the\s+)?(?:one|person|entry|scan|attempt)?\s*"
               r"(?:before|prior to)\s+(?:that|this|him|her|it|them)\b|^\s*(?:and\s+)?(?:the\s+)?previous(?:\s+one)?\s*\??\s*$")

# First match wins; how-to questions skip this list.
DATA_INTENTS = [
    (r"lockdown.*\b(status|active|on|enabled|now)\b|\b(is|are)\b.*\block(ed|down)\b", _lockdown),
    (r"after hours|out of hours|at night|overnight|last night|\b(after|before)\s+\d", _by_time),
    (r"\binactive\b|\b(hasn't|haven't|hasnt|havent|didn't|didnt|not|never)\b.*\b(enter\w*|entries|come|came|scanned|used|visited)\b", _not_in_chat),
    (r"low[- ]?(confidence|match)|weak (match|scan)|\bconfidence\b|borderline|\b(below|under|less than)\s+\d{2}\s*%", _low_match),
    (LAST_DENIED, _last(False)),
    (LAST_GRANTED, _last(True)),
    (r"\b(denied|rejected|refused|failed|unauthori[sz]ed)\b", _count(False)),
    (r"\bhow many\b.*\b(granted|approved|allowed|entries|entered|admitted|came in|got in|let in)\b|\bentries\b|\b(did|has) (anyone|anybody)\b.*\b(enter\w*|come|came|got|get)\b", _count(True)),
    (r"\bhow many\b.*\bscans?\b|\bscans? today\b", _count()),
    (r"\b(how many|number of|count|total)\b.*\b(users?|people|enrolled|students?|faculty|security)\b", _users),
]


# ---- Short-term memory: rewrite a follow-up using the previous question ----

FOLLOW_UP = r"^\s*(?:what about|how about|and what about|and|same for|same but for|what if|also)\b[\s,]*"
HOW_TO = r"\bhow (?:do|can|to|should)\b|\bcan i\b|\bsteps?\b|\bexplain\b"
SUMMARY = r"\b(summari[sz]e|summary|recap|overview|what happened)\b"
PERIOD_PHRASE = (r"\b(?:(?:the\s+)?day before yesterday|\d+\s+(?:day|week|month)s?\s+ago|today|yesterday|last night|overnight|this (?:week|month|year)"
                 r"|(?:the\s+)?(?:past|last|previous|in|within)\s+(?:\d+\s+|(?:" + "|".join(NUMBER_WORDS) + r")\s+)?(?:day|week|month|year)s?"
                 r"|(?:(?:from|since|in|during)\s+)?(?:" + "|".join(MONTHS) + r"|20\d\d)"
                 r"|all time|in total|so far|overall)\b")


FILLER = {"the", "a", "an", "for", "over", "during", "of", "in", "within", "same", "then", "please",
          "about", "it", "that", "those", "them", "ones", "one", "period", "time", "but"}


def _has_intent(q):
    return bool(re.search(SUMMARY, q) or re.search(NAME_QUESTION, q)
                or any(re.search(pattern, q) for pattern, _ in DATA_INTENTS))


UNIT_BEFORE = (r"^\s*(?:and\s+)?(?:what about\s+|how about\s+)?(?:the\s+)?(day|week|month)\s+"
               r"(?:before|prior to)\s+(?:that|this)\b")


def _step_back(unit, previous):
    """'The day before that' after a question about yesterday becomes the same question for 2 days ago."""
    prev = previous.strip(" ?.!")
    low = prev.lower()
    steps = 0
    ago = re.search(r"\b(\d+)\s+" + unit + r"s?\s+ago\b", low)
    if ago:
        steps = int(ago.group(1))
    elif unit == "day" and "day before yesterday" in low:
        steps = 2
    elif unit == "day" and "yesterday" in low:
        steps = 1
    elif unit != "day" and re.search(r"(?<!the )\b(?:last|previous)\s+" + unit + r"\b", low):
        steps = 1
    steps += 1
    named = {"day": "yesterday", "week": "last week", "month": "last month"}
    period = named[unit] if steps == 1 else f"{steps} {unit}s ago"
    return re.sub(r"\s+", " ", re.sub(PERIOD_PHRASE, "", prev, flags=re.I) + " " + period).strip()


def resolve(question, previous):
    """Turn a short follow-up ("what about today?") into a full question, using the previous one."""
    q = (question or "").strip()
    unit = re.search(UNIT_BEFORE, q.lower())
    if unit:
        return _step_back(unit.group(1), previous) if previous else q
    if previous and re.search(BEFORE_THAT, q.lower()):
        prev = previous.lower()
        is_last = (re.search(LAST_GRANTED, prev) or re.search(LAST_DENIED, prev)) and not re.search(
            r"\bwhen did\b|\bdid\s+\w+\s+(?:last\s+)?(?:enter|come)|\bhas\s+\w+\s+(?:entered|come)", prev)
        if is_last:
            n = re.search(r"\b(\d+)(?:st|nd|rd|th) most recent\b", prev)
            n = int(n.group(1)) + 1 if n else 2
            kind = "denied scan" if re.search(LAST_DENIED, prev) else "granted entry"
            return f"{_ordinal(n)} most recent {kind}"
        return q
    m = re.match(FOLLOW_UP, q.lower())
    if not previous or not m:
        return q
    rest = q[m.end():].strip(" ,.?!")
    if not rest or re.search(HOW_TO, rest.lower()):
        return rest or q
    prev = previous.strip(" ?.!")
    rest_period = re.search(PERIOD_PHRASE, rest.lower())
    prev_period = re.search(PERIOD_PHRASE, prev.lower())
    topic = re.sub(PERIOD_PHRASE, "", rest, flags=re.I)
    topic = " ".join(w for w in re.findall(r"[\w'-]+", topic) if w.lower() not in FILLER)
    if rest_period and not topic:                          # only a new period
        return re.sub(r"\s+", " ", re.sub(PERIOD_PHRASE, "", prev, flags=re.I) + " " + rest_period.group()).strip()
    as_count = f"how many {topic}"
    if re.match(r"how many\b", prev.lower()) and not re.match(r"how many\b", topic.lower()) \
            and _has_intent(as_count.lower()):
        topic = as_count                                   # keep the question's form: a count stays a count
    elif not _has_intent(topic.lower()):                   # otherwise a word that is not a topic is a person
        topic = f"when did {topic} last enter"
    period = rest_period.group() if rest_period else (prev_period.group() if prev_period else "")
    return f"{topic} {period}".strip()


NOT_NAMES = {"anyone", "anybody", "someone", "somebody", "everyone", "nobody", "he", "she", "they", "we",
             "i", "you", "it", "the", "a", "an", "any", "people", "user", "users", "student", "students", "staff",
             "never", "not", "ever", "already", "yet", "recently", "just"}


REWRITE_PROMPT = """You turn questions from the admin of AmanGate, a face recognition gate system, into one supported question.
The question may be informal or in Arabic. Reply with exactly one line and nothing else:
- the supported question in English that asks for the same thing, or
- HOWTO if it asks how to do something in the admin portal, or
- NONE if it is not about AmanGate.

Supported questions (fill in the parts in angle brackets):
Summarise <period>
How many granted <period>?
How many denied attempts <period>?
How many scans <period>?
Who entered after <time, e.g. 10 pm> <period>?
Who entered before <time, e.g. 7 am> <period>?
Who entered last night?
Any low-confidence matches <period>?
Who was last granted access?
Who was last denied?
When did <name or ID> last enter?
How many times did <name or ID> enter <period>?
How many <students, faculty, security, visitors or users> are enrolled?
How many <students, faculty, security, visitors or users> were enrolled <period>?
How many people have the first name <name>?
Is lockdown active?

<period> is one of: today, yesterday, this week, last week, this month, last month, this year,
in the past <number> days, in <month name>, in total. Use this week for "recently", "lately" or a habit
("has been coming in"); otherwise use today if no period is given.
Keep names and IDs exactly as written; write Arabic names in English letters.

Examples:
Did anyone sneak in late recently? -> Who entered after 10 pm this week?
Who has been coming in really early? -> Who entered before 7 am this week?
Was anyone flagged as a fake today? -> How many denied attempts today?
كم شخص دخل اليوم؟ -> How many granted today?
كيف أضيف طالب؟ -> HOWTO
What's the weather? -> NONE"""

OUT_OF_SCOPE = ("I can only help with the AmanGate portal and gate activity, for example enrolling or "
                "removing users, locking gates, the dashboard, or questions about past scans.")
DATA_WORDS = r"\b(who|how many|when|any|did|was|were|has|list|show|which|count|number of)\b"


def _data_answer(q, data):
    """Answer from the database if the question matches a known pattern; None otherwise. Never uses the model."""
    ql = q.lower()
    if re.search(SUMMARY, ql):
        return _summary(ql, data)
    if re.search(HOW_TO, ql) or re.search(ACTION_REQUEST, ql):
        return None
    if re.search(NAME_QUESTION, ql) and re.search(r"\b(how many|who|which|list|any|count)\b", ql):
        return _users_named(ql, data)
    if re.search(r"\b(last|when|did|has|times|enter\w*|came|access|granted)\b|\b\d{3,}\b", ql):
        reply = _person(ql, data)
        if reply:
            return reply
        missing_id = re.search(r"\b\d{5,}\b", q)
        if missing_id:
            return f"No enrolled user or recorded scan has the ID {missing_id.group()}."
        asked = re.search(r"\b(?:when did|did|has|last time)\s+([a-z][a-z'-]+)\s+(?:last\s+)?"
                          r"(?:enter|come|came|arrive|scan|get|got|visit|been)", ql)
        if asked and asked.group(1) not in NOT_NAMES:
            return f"I couldn't find anyone called {asked.group(1).title()} among enrolled users or past entries."
    for pattern, handler in DATA_INTENTS:
        if re.search(pattern, ql):
            return handler(ql, data)
    return None


def _rewrite(q):
    """The model turns a free or Arabic question into a supported one; returns the question, 'HOWTO' or 'NONE'."""
    reply = _ask_model(q, REWRITE_PROMPT, max_new_tokens=30).splitlines()
    line = reply[0].strip().strip('"') if reply else "NONE"
    line = re.sub(r"^.*->\s*", "", line).strip()
    return "HOWTO" if line.upper().startswith("HOWTO") else "NONE" if line.upper().startswith("NONE") else line


def answer(question, data):
    """Returns (reply, source, understood); understood is the question actually answered, for follow-ups."""
    q = (question or "").strip()[:300].replace("\u2019", "'")
    if not q:
        return 'Ask me something, e.g. "How do I enroll a student?"', "data", q
    if re.search(UNIT_BEFORE, q.lower()):
        return ("I don't have an earlier question to step back from. Please ask the full question, "
                'e.g. "Summarise yesterday".'), "data", q
    if re.search(BEFORE_THAT, q.lower()):
        return ("I can step back through the most recent entries after a question like "
                '"Who was last granted access?" or "When was the last denied scan?". '
                "For one person, ask e.g. \"How many times did Khaled enter this week?\"."), "data", q
    follow = re.match(FOLLOW_UP, q.lower())
    if follow and not re.search(HOW_TO, q.lower()) and not _has_intent(q.lower()[follow.end():]):
        return ("I don't have an earlier question to follow on from. Please ask the full question, "
                'e.g. "How many denied attempts today?"'), "data", q

    # 1. Known wording: answered from the database at once
    reply = _data_answer(q, data)
    if reply:
        return reply, "data", q

    # 2. A how-to question or command with a clear keyword: the guide section at once
    section = _guide_section(q)
    asks_for_data = re.search(DATA_WORDS, q.lower()) and not re.search(HOW_TO, q.lower()) \
        and not re.search(ACTION_REQUEST, q.lower())
    if section and not asks_for_data:
        return _guide_answer(q, section), "model", q

    # 3. Anything else: the model works out what is being asked
    if _still_loading():
        return ("The AI model is still starting up; please try again in a minute. "
                "Questions about gate activity work now."), "data", q
    if not _load_model():
        return "The AI model is not available. I can still answer questions about gate activity, e.g. \"How many denied attempts this week?\".", "data", q
    try:
        if re.search(HOW_TO, q.lower()) or re.search(ACTION_REQUEST, q.lower()):
            return _guide_answer(q, section), "model", q
        understood = _rewrite(q)
        logging.info("Chat question %r understood as %r", q, understood)
        if understood == "HOWTO":
            return _guide_answer(q, section), "model", q
        if understood != "NONE" and understood.lower() != q.lower():
            reply = _data_answer(understood, data)
            if reply:
                return f"Answering: {understood}\n{reply}", "data", understood
        if section:
            return _guide_answer(q, section), "model", q
        return OUT_OF_SCOPE, "model", q
    except Exception:
        logging.exception("Chat generation failed.")
        return "Sorry, something went wrong. Please try again.", "data", q
