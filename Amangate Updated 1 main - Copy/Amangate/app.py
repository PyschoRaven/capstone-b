from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import base64
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
import threading
import time

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, g, jsonify, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import check_password_hash, generate_password_hash

import chatbot
import recognition


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__, static_folder="static", template_folder="templates")
# No cross-origin calls into this API exist (the terminal/admin pages only
# fetch same-origin paths) — CORS was wide open with no benefit, just attack
# surface. If a real cross-origin client shows up later, scope flask-cors to
# that specific origin instead of reopening it for everyone.
app.secret_key = os.environ.get("SECRET_KEY", "change-me-before-deploying")
if app.secret_key == "change-me-before-deploying":
    logging.warning("Using default SECRET_KEY. Set SECRET_KEY env var before deploying.")
# force=True: importing `recognition` (TensorFlow/absl) attaches its own
# handler to the root logger first, which makes a plain basicConfig() a
# documented no-op — silently leaving the root logger at its default WARNING
# level, so every logging.info() call in this app (scan results, timings,
# gate/lockdown changes, enrollment outcomes) never printed. force=True
# replaces whatever's there so our INFO level actually takes effect.
logging.basicConfig(level=logging.INFO, force=True)

limiter = Limiter(get_remote_address, app=app, default_limits=[])


def scan_rate_key():
    """
    Rate-limit scans per terminal (address + gate), not per address alone:
    through ngrok every iPad arrives from the same address, so one shared
    limit would let a busy gate block the other two.
    """
    gate = (request.get_json(silent=True) or {}).get("gate")
    return f"{get_remote_address()}|{gate if gate in VALID_GATES else 'main'}"


@app.errorhandler(429)
def too_many_requests(_error):
    # JSON (not Flask's HTML page) so the terminal can show a clear message
    return jsonify({"status": "busy", "message": "Too many attempts - please wait a moment"}), 429

# --------------------------
# Configuration
# --------------------------

DB_PATH             = DATA_DIR / "amangate.db"
THRESHOLD           = 0.35
# A frame that already passes ends the scan early. Same decision as checking all
# frames (access uses the best frame, so once one passes the result is GRANTED),
# just without waiting ~1.7 s for each remaining frame.
EARLY_ACCEPT_DIST   = THRESHOLD
ATTENDANCE_LOG_PATH = DATA_DIR / "attendance.csv"
ADMIN_PASSWORD      = os.environ.get("ADMIN_PASSWORD", "admin1234")
VALID_GENDERS       = {"male", "female"}
VALID_GATES         = {"main", "male", "female", "vehicle"}
VALID_ROLES         = {"student", "faculty", "security", "visitor"}
VISITOR_TTL         = timedelta(hours=24)
lockdown_enabled    = False

if ADMIN_PASSWORD == "admin1234":
    logging.warning("Using default admin password. Set ADMIN_PASSWORD env var before deploying.")

# Compared with check_password_hash() (constant-time) instead of the raw
# string, so a login never leaks timing info about the password's prefix.
ADMIN_PASSWORD_HASH = generate_password_hash(ADMIN_PASSWORD)

# Encrypts enrolled face embeddings at rest, so a copy of amangate.db on its
# own doesn't hand over usable biometric data. EMBED_KEY can be any string —
# it's stretched into a 32-byte Fernet key below — but losing/changing it
# after enrolling users makes their stored embeddings unreadable (they would
# need to be re-enrolled), so treat it like any other secret you must keep.
EMBED_KEY = os.environ.get("EMBED_KEY", "change-me-before-deploying")
if EMBED_KEY == "change-me-before-deploying":
    logging.warning("Using default EMBED_KEY. Set EMBED_KEY env var before deploying "
                     "— otherwise enrolled face data is only protected by a publicly-known key.")
_fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(EMBED_KEY.encode("utf-8")).digest()))


def error_response(message="Something went wrong. Please try again."):
    return jsonify({"status": "error", "message": message})


# --------------------------
# Auth
# --------------------------

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["POST"])
@limiter.limit("5 per minute")
def admin_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username") or ""
    password = data.get("password") or ""
    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify(status="error", message="Incorrect username or password."), 401
    username = username.strip()
    valid = (check_password_hash(ADMIN_PASSWORD_HASH, password) if username.lower() in {"", "admin"}
             else authenticate_admin_account(username, password))
    if valid:
        session["admin"] = True
        session["admin_username"] = username or "admin"
        session.pop("admin_account_csrf", None)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Incorrect username or password. Please try again."}), 401


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin", None)
    session.pop("admin_username", None)
    session.pop("admin_account_csrf", None)
    session.pop("chat_last", None)
    return jsonify({"status": "success"})


@app.route("/admin/session")
def admin_session():
    return jsonify({"authenticated": bool(session.get("admin"))})


# --------------------------
# Database helpers
# --------------------------

def get_db():
    """Open and return a SQLite connection with row factory enabled."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def dump_embedding(embedding):
    """Serialize + encrypt a face embedding (or list of per-view embeddings)."""
    return _fernet.encrypt(json.dumps(embedding).encode("utf-8"))


class EmbeddingUnreadable(Exception):
    """A stored face was encrypted with a different EMBED_KEY (or is damaged)."""


def _is_fernet_token(blob):
    # Fernet tokens are url-safe base64 of a version byte 0x80 -> always start "gAAAAA"
    return bytes(blob[:6]) == b"gAAAAA"


def load_embedding(blob):
    """
    Decrypt + parse a stored embedding. Rows written by this version are
    Fernet-encrypted JSON. Older rows were plain JSON, and before that,
    pickled; migrate_embeddings_to_encrypted() upgrades everything to
    encrypted JSON once at startup, so the fallbacks below only ever run
    against this app's own pre-existing local data, never anything written
    afterwards.

    Raises EmbeddingUnreadable when the row is encrypted with a different
    EMBED_KEY — never tries to unpickle an encrypted token.
    """
    blob = bytes(blob)
    try:
        return json.loads(_fernet.decrypt(blob).decode("utf-8"))
    except InvalidToken:
        pass
    if _is_fernet_token(blob):
        raise EmbeddingUnreadable("encrypted with a different EMBED_KEY")
    if blob[:1] == b"[":
        return json.loads(blob.decode("utf-8"))
    try:
        return pickle.loads(blob)
    except Exception as e:
        raise EmbeddingUnreadable(f"unreadable face data ({e.__class__.__name__})") from None


def migrate_embeddings_to_encrypted(conn):
    """
    One-time upgrade: encrypt any embedding not already in the current format.
    A row that can't be read (e.g. EMBED_KEY was changed) is left as it is and
    reported — it must never stop the server from starting.
    """
    rows = conn.execute("SELECT id, name, embedding FROM users").fetchall()
    migrated = 0
    for row in rows:
        blob = bytes(row["embedding"])
        try:
            _fernet.decrypt(blob)
            continue  # already encrypted with the current EMBED_KEY
        except InvalidToken:
            pass
        try:
            embedding = load_embedding(blob)
        except EmbeddingUnreadable as e:
            logging.error("Face data of %s (%s) is %s — re-register this person.",
                          row["name"], row["id"], e)
            continue
        conn.execute("UPDATE users SET embedding = ? WHERE id = ?",
                     (dump_embedding(embedding), row["id"]))
        migrated += 1
    if migrated:
        conn.commit()
        logging.info("Encrypted %d embedding(s) at rest.", migrated)


_users_cache      = None
_users_cache_lock = threading.Lock()


def get_cached_users():
    """Enrolled users + embeddings, loaded once and reused across scans."""
    global _users_cache
    with _users_cache_lock:
        if _users_cache is None:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT id, name, role, gates, embedding FROM users"
                ).fetchall()
            users = []
            for row in rows:
                try:
                    users.append((row, load_embedding(row["embedding"])))
                except EmbeddingUnreadable as e:
                    # Skip this one person instead of failing every scan for everyone
                    logging.error("Skipping %s (%s) in scans: face data %s — re-register them.",
                                  row["name"], row["id"], e)
            _users_cache = users
        return _users_cache


def invalidate_users_cache():
    global _users_cache
    with _users_cache_lock:
        _users_cache = None


def init_db():
    """Create tables if they don't exist and load persisted state."""
    global lockdown_enabled
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                role        TEXT DEFAULT 'student',
                embedding   BLOB NOT NULL,
                enrolled_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # History for the System Analytics page
        conn.execute("CREATE TABLE IF NOT EXISTS lockdown_events (at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS scan_times (at TEXT NOT NULL, ms REAL NOT NULL)")
        # Every scan, linked to the user's ID (NULL for unknown faces and spoof attempts)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                user_id    TEXT,
                name       TEXT NOT NULL,
                confidence REAL,
                status     TEXT NOT NULL
            )
        """)
        upgrade_scans_table(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS scans_by_time ON scans (scanned_at)")
        # Columns added after the first release — add them to existing databases
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        for column in ("gender", "gates", "expires_at"):
            if column not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT")
        # Gate types are fixed now (separate Vehicle Gate) — drop the old per-gate switch
        conn.execute("DELETE FROM settings WHERE key LIKE 'gate_type:%'")
        conn.commit()
        migrate_embeddings_to_encrypted(conn)
        import_csv_log(conn)

    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = 'lockdown'").fetchone()
        lockdown_enabled = (row["value"] == "1") if row else False
    logging.info("Lockdown state loaded from DB: %s", lockdown_enabled)


def purge_expired_visitors(conn):
    """Delete visitors whose 24-hour access has run out."""
    now = datetime.now().isoformat(timespec="seconds")
    result = conn.execute(
        "DELETE FROM users WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
    )
    if result.rowcount:
        conn.commit()
        invalidate_users_cache()
        logging.info("Removed %d expired visitor(s).", result.rowcount)


VISITOR_CLEANUP_SECONDS = 60


def start_visitor_cleanup():
    """
    Remove expired visitors on a timer, so they leave the system by themselves
    even when nobody is scanning or using the admin portal.
    """
    def loop():
        while True:
            try:
                with get_db() as conn:
                    purge_expired_visitors(conn)
            except Exception:
                logging.exception("Visitor cleanup failed.")
            time.sleep(VISITOR_CLEANUP_SECONDS)

    threading.Thread(target=loop, name="visitor-cleanup", daemon=True).start()
    logging.info("Visitor cleanup running every %d seconds.", VISITOR_CLEANUP_SECONDS)


# --------------------------
# Attendance logging
# --------------------------

def log_attendance_event(user_id, name, confidence_score, status):
    """
    Record a scan in the scans table. user_id is the enrolled user's ID, or None
    for unknown faces and spoof attempts. The name is kept as it was at scan time,
    so scans of removed or expired users stay readable.
    """
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scans (scanned_at, user_id, name, confidence, status) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, name, confidence_score, status),
        )
        conn.commit()


def upgrade_scans_table(conn):
    """
    An earlier test version stored scans with a granted (0/1) column instead of a
    status. Rebuild that table once with a status column, keeping every scan.
    """
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
    if "status" in columns or "granted" not in columns:
        return
    conn.execute("ALTER TABLE scans RENAME TO scans_old")
    conn.execute("""
        CREATE TABLE scans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL,
            user_id    TEXT,
            name       TEXT NOT NULL,
            confidence REAL,
            status     TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO scans (id, scanned_at, user_id, name, confidence, status)
        SELECT id, scanned_at, user_id, name, confidence,
               CASE WHEN granted = 1 THEN 'GRANTED' ELSE 'DENIED' END
        FROM scans_old
    """)
    conn.execute("DROP TABLE scans_old")
    conn.execute("DROP INDEX IF EXISTS scans_by_time")
    conn.commit()
    logging.info("Upgraded the scans table to store each scan's status.")


def import_csv_log(conn):
    """
    One-time import of the old data/attendance.csv into the scans table. A name is
    linked to a user ID when exactly one enrolled user has that name. The CSV is
    left in place as a backup and is no longer written to.
    """
    if conn.execute("SELECT 1 FROM settings WHERE key = 'csv_imported'").fetchone():
        return
    imported = 0
    if os.path.exists(ATTENDANCE_LOG_PATH):
        ids = {}
        for r in conn.execute("SELECT id, name FROM users"):
            ids.setdefault(r["name"], []).append(r["id"])
        with open(ATTENDANCE_LOG_PATH, encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4:
                    continue
                try:
                    at = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                    confidence = float(parts[-2].rstrip("%"))
                except ValueError:
                    continue
                name   = ",".join(parts[1:-2])  # names were written unquoted
                status = parts[-1].strip()
                matches = ids.get(name, []) if status in ("GRANTED", "DENIED_GATE") else []
                conn.execute(
                    "INSERT INTO scans (scanned_at, user_id, name, confidence, status) VALUES (?, ?, ?, ?, ?)",
                    (at.strftime("%Y-%m-%d %H:%M:%S"), matches[0] if len(matches) == 1 else None,
                     name, confidence, status),
                )
                imported += 1
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('csv_imported', '1')")
    conn.commit()
    if imported:
        logging.info("Imported %d scans from %s into the database.", imported, ATTENDANCE_LOG_PATH.name)


def record_scan_time(started):
    """Store how long a full recognition scan took (seconds since `started`, as ms)."""
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO scan_times (at, ms) VALUES (?, ?)",
                         (datetime.now().isoformat(timespec="seconds"), (time.perf_counter() - started) * 1000))
            conn.commit()
    except Exception:
        logging.exception("Could not record scan time.")


def read_attendance_log():
    """Return every scan, oldest first. Scans linked to a user show their current name."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.scanned_at, s.user_id, COALESCE(u.name, s.name) AS name, s.confidence, s.status
            FROM scans s LEFT JOIN users u ON u.id = s.user_id
            ORDER BY s.scanned_at, s.id
        """).fetchall()
    return [{
        "at":         datetime.strptime(r["scanned_at"], "%Y-%m-%d %H:%M:%S"),
        "user_id":    r["user_id"],
        "name":       r["name"],
        "confidence": r["confidence"] or 0.0,
        "granted":    r["status"] == "GRANTED",
        # DENIED (unknown face), DENIED_GATE (known, wrong gate), DENIED_SPOOF
        "status":     r["status"],
    } for r in rows]


def build_scan_alerts(denied_events):
    """
    Turn denied scans into alerts. Three or more denials within five minutes
    of each other become one "Multiple failed scans" alert; anything else is
    reported as a single unauthorized attempt.
    """
    alerts, group = [], []

    def describe(e):
        if e.get("status") == "DENIED_GATE":
            return "Wrong gate", f"{e['name']} is not permitted at this gate"
        if e.get("status") == "DENIED_SPOOF":
            return "Spoof attempt", "Screen or printed photo detected"
        return "Unauthorized access attempt", "Unknown face denied"

    def flush():
        if len(group) >= 3:
            alerts.append({"level": "yellow", "title": "Multiple failed scans",
                           "detail": f"{len(group)} denied attempts", "at": group[-1]["at"]})
        else:
            for e in group:
                title, detail = describe(e)
                alerts.append({"level": "red", "title": title, "detail": detail, "at": e["at"]})
        group.clear()

    for e in denied_events:
        if group and (e["at"] - group[-1]["at"]).total_seconds() > 300:
            flush()
        group.append(e)
    flush()
    return sorted(alerts, key=lambda a: a["at"], reverse=True)


# --------------------------
# Dashboard stats (real data from the attendance log)
# --------------------------

@app.route("/dashboard/stats")
@admin_required
def dashboard_stats():
    try:
        events    = read_attendance_log()
        now       = datetime.now()
        today     = now.date()
        yesterday = today - timedelta(days=1)

        todays  = [e for e in events if e["at"].date() == today]
        granted = sum(1 for e in todays if e["granted"])

        # Granted entries per hour for the last 12 hours (current hour last)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        hourly = []
        for i in range(11, -1, -1):
            start = current_hour - timedelta(hours=i)
            end   = start + timedelta(hours=1)
            hourly.append({
                "hour":  start.isoformat(timespec="minutes"),
                "count": sum(1 for e in events if e["granted"] and start <= e["at"] < end),
            })

        def serialize(e):
            return {"at": e["at"].isoformat(timespec="seconds"), "name": e["name"],
                    "confidence": e["confidence"], "granted": e["granted"]}

        alerts = build_scan_alerts([e for e in todays if not e["granted"]])

        return jsonify({
            "status":    "success",
            "now":       now.isoformat(timespec="seconds"),
            "today":     {"granted": granted, "denied": len(todays) - granted, "total": len(todays)},
            "yesterday": {"granted": sum(1 for e in events if e["granted"] and e["at"].date() == yesterday)},
            "hourly":    hourly,
            "recent":    [serialize(e) for e in reversed(events[-7:])],
            "last_scan": events[-1]["at"].isoformat(timespec="seconds") if events else None,
            "alerts":    [{**a, "at": a["at"].isoformat(timespec="seconds")} for a in alerts],
        })
    except Exception:
        logging.exception("Dashboard stats failed.")
        return error_response("Could not load dashboard data.")


# --------------------------
# System overview stats (users + last 7 days of scans)
# --------------------------

USER_GROUPS = {"student": "students", "faculty": "employees", "security": "employees", "visitor": "visitors"}


def _avg(values):
    return round(sum(values) / len(values), 1) if values else None


@app.route("/system/stats")
@admin_required
def system_stats():
    """Everything the System Analytics page shows: this week vs. last week."""
    try:
        now        = datetime.now()
        today      = now.date()
        week_start = today - timedelta(days=6)    # this week = the last 7 days, today included
        prev_start = week_start - timedelta(days=7)

        def week_of(d):
            if d >= week_start: return "this"
            if d >= prev_start: return "prev"
            return None

        with get_db() as conn:
            purge_expired_visitors(conn)
            users     = conn.execute("SELECT role, enrolled_at FROM users").fetchall()
            lockdowns = [datetime.fromisoformat(r["at"]) for r in conn.execute("SELECT at FROM lockdown_events")]
            timings   = [(datetime.fromisoformat(r["at"]), r["ms"]) for r in conn.execute("SELECT at, ms FROM scan_times")]

        events = read_attendance_log()
        weeks  = {"this": [], "prev": []}
        for e in events:
            w = week_of(e["at"].date())
            if w:
                weeks[w].append(e)

        def summary(evts, w):
            granted = [e for e in evts if e["granted"]]
            ms      = [m for at, m in timings if week_of(at.date()) == w]
            return {
                "granted":        len(granted),
                "denied":         len(evts) - len(granted),
                "avg_confidence": _avg([e["confidence"] for e in granted]),
                "lockdowns":      sum(1 for at in lockdowns if week_of(at.date()) == w),
                "avg_scan_ms":    _avg(ms),
            }

        # Granted entries per day, oldest first
        days = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            days.append({"date": day.isoformat(),
                         "granted": sum(1 for e in weeks["this"] if e["granted"] and e["at"].date() == day)})

        # Match confidence of granted scans, last 30 days, in 5-point bins from the threshold up
        floor = int((1 - THRESHOLD) * 100) // 5 * 5
        bins  = [{"from": b, "to": b + 5, "count": 0} for b in range(floor, 100, 5)]
        for e in events:
            if e["granted"] and e["at"].date() > today - timedelta(days=30):
                idx = min(len(bins) - 1, max(0, int((e["confidence"] - floor) // 5)))
                bins[idx]["count"] += 1

        # Enrolled users at the end of each of the last 8 weeks (current users only)
        enrolled = sorted(datetime.fromisoformat(u["enrolled_at"][:19]) for u in users if u["enrolled_at"])
        growth = []
        for i in range(7, -1, -1):
            cutoff = now - timedelta(weeks=i)
            growth.append({"week_end": cutoff.date().isoformat(), "count": sum(1 for d in enrolled if d <= cutoff)})
        month_ago = now - timedelta(days=30)

        # Every scan by hour of day, this week combined
        hours = [0] * 24
        for e in weeks["this"]:
            hours[e["at"].hour] += 1

        groups = {"students": 0, "employees": 0, "visitors": 0}
        for u in users:
            groups[USER_GROUPS.get(u["role"], "students")] += 1

        return jsonify({
            "status":     "success",
            "now":        now.isoformat(timespec="seconds"),
            "threshold":  round((1 - THRESHOLD) * 100, 1),
            "this_week":  summary(weeks["this"], "this"),
            "last_week":  summary(weeks["prev"], "prev"),
            "days":       days,
            "confidence": bins,
            "growth":     growth,
            "new_this_month": sum(1 for d in enrolled if d > month_ago),
            "hours":      hours,
            "groups":     groups,
            "users_total": len(users),
        })
    except Exception:
        logging.exception("System stats failed.")
        return error_response("Could not load system data.")


# --------------------------
# Page routes
# --------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/system")
def system_overview():
    return render_template("system.html")


# --------------------------
# Face centering check route
# --------------------------

@app.route("/check-center", methods=["POST"])
def check_center():
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image")
        if not image_data:
            return jsonify({"centered": False, "reason": "error", "message": "No image provided"})
        with get_db() as conn:
            vehicle = gate_info(conn, gate_from_request(data))["type"] == "vehicle"
        return jsonify(recognition.check_face_center(image_data, vehicle=vehicle))
    except Exception:
        logging.exception("Center check failed.")
        return jsonify({
            "centered": False,
            "reason": "error",
            "message": "Could not check face position. Please try again.",
        })


# --------------------------
# Scan route
# --------------------------

@app.route("/scan", methods=["POST"])
# A scan needs a fresh blink and shows each result for a few seconds, so a busy
# gate does at most ~10-15 scans a minute; 60 leaves room without allowing floods.
@limiter.limit("60 per minute", key_func=scan_rate_key)
def scan_face():
    started = time.perf_counter()
    try:
        if lockdown_enabled:
            return jsonify({"status": "lockdown", "message": "Emergency Lockdown Active"})

        data = request.get_json(silent=True) or {}
        images = data.get("images")
        if not images:
            return error_response("No image received. Please try again.")
        # No Haar Cascade check here: RetinaFace below decides whether there is
        # a face (and only one), so angled / dim faces are not turned away early.
        gate = gate_from_request(data)
    except Exception:
        logging.exception("Scan setup failed.")
        return error_response("Scan failed. Please try again.")

    try:
        with get_db() as conn:
            purge_expired_visitors(conn)
            gate_state = gate_info(conn, gate)
        users = get_cached_users()
    except Exception:
        logging.exception("Could not load users for scan.")
        return error_response("Scan failed. Please try again.")

    if gate_state["locked"]:
        return jsonify({"status": "gate_locked", "message": f"{gate_state['name']} is closed"})
    # Vehicle gates: use the driver's face (the largest) and ignore passengers
    vehicle = gate_state["type"] == "vehicle"

    best_match      = None
    best_dist       = 1.0
    best_confidence = 0.0

    try:
        current_embeddings = []
        face_box   = None
        problem    = None   # first reason a frame was not usable (shown as guidance)
        yaws       = []
        best_frame = None   # raw frame + box for the best-matching face seen so
        best_box   = None   # far — anti-spoofing runs once against these, below
        # Frames are checked one at a time (each takes ~1.7 s on this CPU).
        # As soon as one frame is a clear match, the rest are skipped — a clear
        # face is granted after one frame instead of three. Anti-spoofing is
        # skipped per-frame here and run once below, only on the winning frame.
        for img in images:
            frame_started = time.perf_counter()
            try:
                face = recognition.analyze_face(img, vehicle=vehicle, run_spoof=False)
                logging.info("Scan timing: frame %d took %.0f ms (detail: %s)", len(yaws) + 1,
                             (time.perf_counter() - frame_started) * 1000, face.get("timing"))
            except recognition.MultipleFacesError:
                logging.info("Scan stopped: RetinaFace found more than one person.")
                return jsonify({"status": "multiple_faces",
                                "message": "Multiple faces detected - one person at a time"})
            except recognition.FaceCheckError as e:
                problem = problem or (e.reason, e.message)
                continue
            except Exception:
                continue  # skip frames where no face was detected

            # Head turned too far: guide the person instead of matching a bad angle
            if abs(face["yaw"]) > recognition.YAW_SCAN_MAX:
                problem = problem or ("turned", recognition.turn_back_message(face["yaw"]))
                continue
            current_embeddings.append(face["embedding"])
            yaws.append(face["yaw"])
            face_box = face["box"]  # keep the latest box for UI overlay
            g.scan_capture = (face["frame"], face["box"])

            for row, stored in users:
                dist, confidence = recognition.calculate_match_multi([face["embedding"]], stored)
                if dist < best_dist:
                    best_dist, best_match, best_confidence = dist, row, confidence
                    best_frame, best_box, best_emb = face["frame"], face["box"], face["embedding"]
            if best_dist < EARLY_ACCEPT_DIST:
                break  # clear match — no need to check the remaining frames

        if not current_embeddings:
            if problem:
                # Not a denial — nothing is logged, the terminal shows the guidance
                logging.info("Scan needs adjustment: %s", problem[0])
                return jsonify({"status": "adjust", "reason": problem[0], "message": problem[1]})
            logging.info("Face embedding failed for all frames during scan.")
            return jsonify({"status": "no_face", "message": "No face detected"})
    except Exception:
        logging.exception("Face embedding failed during scan.")
        return jsonify({"status": "no_face", "message": "No face detected"})

    # Anti-spoofing: run once, against the frame that produced the best match,
    # instead of on every captured frame (was the main per-scan cost besides
    # RetinaFace/ArcFace itself).
    if best_frame is not None:
        g.scan_capture = (best_frame, best_box)
    spoof = None
    g.security_spoof_check = "disabled" if recognition.ANTI_SPOOF_MODE == "off" else "not_run"
    if best_frame is not None and recognition.ANTI_SPOOF_MODE != "off":
        try:
            spoof = recognition.check_spoof(best_frame, best_box)
            g.security_spoof_check = "passed" if spoof[0] else "flagged"
        except Exception:
            g.security_spoof_check = "error"
            logging.exception("Anti-spoofing check failed.")
        if recognition.ANTI_SPOOF_MODE == "enforce" and spoof and not spoof[0]:
            logging.info("Scan result: SPOOF detected - access denied.")
            log_attendance_event(None, "Spoof attempt", 0, "DENIED_SPOOF")
            return jsonify({"status": "spoof", "message": "Fake face detected - screen or photo"})

    try:
        # ----------------------------------------------------------------
        # Logging only — records the winning distance/confidence for this
        # scan attempt, for later threshold tuning / accuracy analysis.
        # ----------------------------------------------------------------
        recognized = best_match is not None and best_dist < THRESHOLD
        # Gate permission: the gates ticked for this person when they were added.
        # People enrolled before gate access existed (nothing ticked) may use any gate.
        allowed_gates = [g for g in ((best_match["gates"] if best_match else "") or "").split(",") if g]
        permitted = recognized and (not allowed_gates or gate in allowed_gates)

        logging.info(
            "Scan result: gate=%s%s user=%s distance=%.4f confidence=%.1f%% yaw=%s spoof(real,score)=%s result=%s",
            gate, " (vehicle)" if vehicle else "",
            best_match["name"] if best_match else "Unknown",
            best_dist, best_confidence, yaws, spoof,
            "GRANTED" if permitted else ("DENIED (not permitted at this gate)" if recognized else "DENIED"),
        )

        record_scan_time(started)
        with get_db() as conn:
            set_setting(conn, f"gate_last_scan:{gate}", datetime.now().isoformat(timespec="seconds"))
            conn.commit()

        # Temporary: confirm the faster detection gives the same result as the
        # previous one — runs after the reply is sent, logs numbers only.
        if recognition.VERIFY_DETECTION_SPEEDUP and best_frame is not None and best_match is not None:
            stored_me = next((v for row, v in users if row["id"] == best_match["id"]), None)

            def verify(frame=best_frame, emb=best_emb, box=best_box, stored=stored_me, new_conf=best_confidence):
                try:
                    diff, prev_emb, box_diff = recognition.compare_with_previous_detection(frame, emb, box)
                    _, prev_conf = recognition.calculate_match_multi([prev_emb], stored)
                    logging.info("Detection check: new vs previous embedding diff=%.4f box diff=%dpx "
                                 "match new=%.1f%% previous=%.1f%%", diff, box_diff, new_conf, prev_conf)
                except Exception:
                    logging.exception("Detection check failed.")
            threading.Thread(target=verify, daemon=True).start()

        if recognized and not permitted:
            log_attendance_event(best_match["id"], best_match["name"], best_confidence, "DENIED_GATE")
            return jsonify({
                "status":  "not_permitted",
                "user":    best_match["name"],
                "message": f"Not permitted at the {gate_state['name']}",
            })

        if recognized:
            log_attendance_event(best_match["id"], best_match["name"], best_confidence, "GRANTED")
            return jsonify({
                "status":     "success",
                "user":       best_match["name"],
                "role":       best_match["role"].capitalize(),
                "confidence": best_confidence,
                "box":        face_box,
            })

        log_attendance_event(None, "Unknown", best_confidence, "DENIED")
        return jsonify({
            "status":     "denied",
            "message":    "Unknown User",
            "confidence": best_confidence,
            "box":        face_box,
        })
    except Exception:
        logging.exception("Scan failed.")
        return error_response("Scan failed. Please try again.")


# --------------------------
# Enroll route
# --------------------------

ENROLL_POSES = {
    "straight": "Photo 1 (look straight)",
    "left":     "Photo 2 (turn slightly left)",
    "right":    "Photo 3 (turn slightly right)",
}
# The three photos must show the same person (cosine distance to the straight view)
SAME_PERSON_MAX_DIST = 0.62   # turned views differ naturally (seen: 0.40-0.56); other people are much further apart


def enroll_views(views):
    """
    Check the three guided photos and return (list_of_embeddings, None),
    or (None, message) saying which photo to retake and why.
    Each view is kept separately, so a scan is matched against the closest angle.
    """
    by_pose = {v.get("pose"): v.get("image") for v in views if v.get("pose") in ENROLL_POSES}
    if set(by_pose) != set(ENROLL_POSES) or not all(by_pose.values()):
        return None, "Three photos are needed: straight, slightly left and slightly right."

    embeddings = {}
    for pose, label in ENROLL_POSES.items():
        try:
            face = recognition.analyze_face(by_pose[pose])
        except recognition.MultipleFacesError:
            return None, f"{label}: more than one person is in the photo."
        except recognition.FaceCheckError as e:
            return None, f"{label}: {e.message}."
        except recognition.SpoofError:
            return None, f"{label}: this does not look like a real face (screen or printed photo)."
        except Exception:
            return None, f"{label}: no face found. Please retake."

        yaw = face["yaw"]
        logging.info("Enrollment %s photo: yaw=%.2f spoof(real,score)=%s", pose, yaw, face["spoof"])
        if pose == "straight" and abs(yaw) > recognition.YAW_STRAIGHT:
            return None, f"{label}: please look straight at the camera."
        # Side photos: only the amount of turn is checked here. Which way is
        # checked below — people easily mix up left and right, so either order is fine.
        if pose != "straight":
            if abs(yaw) < recognition.YAW_SIDE_MIN:
                return None, f"{label}: turn your head a little more to the side."
            if abs(yaw) > recognition.YAW_SIDE_MAX:
                return None, f"{label}: turned too far - just a slight turn."
        embeddings[pose] = face

    if embeddings["left"]["yaw"] * embeddings["right"]["yaw"] > 0:
        return None, "Photos 2 and 3 were turned the same way - turn one way for photo 2 and the other way for photo 3."

    for pose in ("left", "right"):
        dist, _ = recognition.calculate_match(embeddings[pose]["embedding"], embeddings["straight"]["embedding"])
        logging.info("Enrollment %s photo: distance to straight photo=%.3f", pose, dist)
        if dist > SAME_PERSON_MAX_DIST:
            return None, f"{ENROLL_POSES[pose]} does not look like the same person as photo 1. Please retake."

    return [list(embeddings[p]["embedding"]) for p in ENROLL_POSES], None


@app.route("/enroll", methods=["POST"])
@limiter.limit("20 per minute")
@admin_required
def enroll_user():
    try:
        data       = request.get_json(silent=True) or {}
        uid        = data.get("id", "").strip()
        name       = data.get("name", "").strip().title()
        role       = data.get("role", "student")
        gender     = data.get("gender") if data.get("gender") in VALID_GENDERS else None
        gates      = ",".join(g for g in (data.get("gates") or []) if g in VALID_GATES)
        # Guided enrollment sends "views": straight, slightly left, slightly right.
        # Older clients send a burst of frames ("images") or a single frame ("image").
        views  = data.get("views") or []
        images = data.get("images") or ([data["image"]] if data.get("image") else [])

        if not uid or not name:
            return error_response("ID and Name are both required")
        if not uid.isdigit():
            return error_response("ID must contain numbers only")
        if role not in VALID_ROLES:
            return error_response("Please choose a valid role")
        if not views and not images:
            return error_response("No photos received. Please capture face photos first.")

        if views:
            embedding, problem = enroll_views(views)
            if problem:
                logging.info("Enrollment rejected: %s", problem)
                return error_response(problem)
        else:
            embeddings = []
            for img in images:
                try:
                    embeddings.append(recognition.analyze_face(img, check=False)["embedding"])
                except recognition.MultipleFacesError:
                    return error_response("More than one person is in the photo. Please make sure only one face is visible.")
                except Exception:
                    continue  # skip frames where no face was detected
            if not embeddings:
                return error_response("No clear face found in the photo. Please retake it.")
            # Average the frames into one embedding — steadier than a single shot
            embedding = np.mean(embeddings, axis=0).tolist()

        now = datetime.now()
        # Visitors get access for 24 hours only; everyone else never expires
        expires_at = (now + VISITOR_TTL).isoformat(timespec="seconds") if role == "visitor" else None

        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO users (id, name, role, embedding, enrolled_at, gender, gates, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (uid, name, role, dump_embedding(embedding),
                  now.isoformat(timespec="seconds"), gender, gates, expires_at))
            conn.commit()
        invalidate_users_cache()

        message = f"{name} enrolled successfully!"
        if expires_at:
            message = f"{name} registered as a visitor until {(now + VISITOR_TTL).strftime('%d %b, %H:%M')}."
        return jsonify({"status": "success", "message": message})
    except Exception:
        logging.exception("Enrollment failed.")
        return error_response("Enrollment failed. Please make sure one clear face is visible.")


# --------------------------
# User management routes
# --------------------------

@app.route("/users", methods=["GET"])
@admin_required
def list_users():
    """Return enrolled users without their embeddings."""
    try:
        with get_db() as conn:
            purge_expired_visitors(conn)
            rows = conn.execute(
                "SELECT id, name, role, gender, gates, enrolled_at, expires_at FROM users ORDER BY name"
            ).fetchall()
        users = []
        for r in rows:
            user = dict(r)
            user["enrolled_at"] = (user["enrolled_at"] or "")[:19] or None  # drop microseconds
            user["gates"] = [g for g in (user["gates"] or "").split(",") if g]
            users.append(user)
        return jsonify({"status": "success", "users": users})
    except Exception:
        logging.exception("User list failed.")
        return error_response("Could not load users. Please try again.")


@app.route("/delete_user", methods=["POST"])
@limiter.limit("20 per minute")
@admin_required
def delete_user():
    """Remove a user by their unique ID."""
    try:
        data = request.get_json(silent=True) or {}
        uid  = data.get("id", "").strip()
        if not uid:
            return error_response("No ID provided")

        with get_db() as conn:
            result = conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            conn.commit()

        if result.rowcount == 0:
            return error_response(f"ID '{uid}' not found")

        invalidate_users_cache()
        return jsonify({"status": "success", "message": "User removed successfully."})
    except Exception:
        logging.exception("Delete user failed.")
        return error_response("Could not remove user. Please try again.")


# --------------------------
# Lockdown routes
# --------------------------

@app.route("/lockdown/status")
def lockdown_status():
    return jsonify({"lockdown": lockdown_enabled})


@app.route("/lockdown/on", methods=["POST"])
@limiter.limit("10 per minute")
@admin_required
def lockdown_on():
    global lockdown_enabled
    lockdown_enabled = True
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('lockdown', '1')")
        conn.execute("INSERT INTO lockdown_events (at) VALUES (?)", (datetime.now().isoformat(timespec="seconds"),))
        conn.commit()
    return jsonify({"lockdown": True, "message": "Emergency Lockdown Enabled"})


@app.route("/lockdown/off", methods=["POST"])
@limiter.limit("10 per minute")
@admin_required
def lockdown_off():
    global lockdown_enabled
    lockdown_enabled = False
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('lockdown', '0')")
        conn.commit()
    return jsonify({"lockdown": False, "message": "Lockdown Disabled - Terminal Ready"})


# --------------------------
# Gates — one server, one terminal (iPad) per gate
# --------------------------
# Each gate has its own lock and last-scan time, saved in the settings table so
# every terminal and the admin dashboard see the same state. A gate's type is
# fixed by what the gate is:
#   "walk"    — Main, Male and Female Gate: people on foot (one person in the frame)
#   "vehicle" — Vehicle Gate: drivers in cars — only the driver's (largest) face is
#               used, passengers are ignored, and a smaller face is accepted

GATE_NAMES = {"main": "Main Gate", "male": "Male Gate", "female": "Female Gate", "vehicle": "Vehicle Gate"}
GATE_TYPES = {"main": "walk", "male": "walk", "female": "walk", "vehicle": "vehicle"}


def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def gate_info(conn, gate_id):
    return {
        "id":        gate_id,
        "name":      GATE_NAMES[gate_id],
        "locked":    get_setting(conn, f"gate_locked:{gate_id}") == "1",
        "type":      GATE_TYPES[gate_id],
        "last_scan": get_setting(conn, f"gate_last_scan:{gate_id}"),
    }


def gate_from_request(data):
    """The gate a terminal says it is; unknown or missing -> Main Gate."""
    gate = (data or {}).get("gate")
    return gate if gate in GATE_NAMES else "main"


@app.route("/gates")
def list_gates():
    """Gate state for the terminals (lock + type) and the admin dashboard."""
    with get_db() as conn:
        return jsonify({"status": "success", "lockdown": lockdown_enabled,
                        "gates": [gate_info(conn, g) for g in GATE_NAMES]})


@app.route("/gates/<gate_id>", methods=["POST"])
@limiter.limit("30 per minute")
@admin_required
def update_gate(gate_id):
    """Lock / unlock one gate (its walk-in / vehicle type is fixed)."""
    if gate_id not in GATE_NAMES:
        return error_response("Unknown gate"), 404
    data = request.get_json(silent=True) or {}
    with get_db() as conn:
        if "locked" in data:
            set_setting(conn, f"gate_locked:{gate_id}", "1" if data["locked"] else "0")
        conn.commit()
        info = gate_info(conn, gate_id)
    logging.info("Gate updated: %s locked=%s type=%s", gate_id, info["locked"], info["type"])
    return jsonify({"status": "success", "gate": info})


# --------------------------
# Admin assistant (chatbot)
# --------------------------

def chat_user_counts():
    with get_db() as conn:
        rows = conn.execute("SELECT role, COUNT(*) AS n FROM users GROUP BY role").fetchall()
    return {(r["role"] or "student"): r["n"] for r in rows}


def chat_users():
    with get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT id, name, role, enrolled_at FROM users")]


@app.route("/chat", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()[:300]
        question = chatbot.resolve(message, session.get("chat_last"))
        reply, source, understood = chatbot.answer(question, {
            "user_counts": chat_user_counts,
            "users":       chat_users,
            "events":      read_attendance_log,
            "lockdown":    lambda: lockdown_enabled,
        })
        if not re.match(chatbot.FOLLOW_UP, understood.lower()):
            session["chat_last"] = understood         # short-term memory, cleared at logout
        if question.lower() not in message.lower():
            logging.info("Chat follow-up %r read as %r", message, question)
        return jsonify({"status": "success", "answer": reply, "source": source})
    except Exception:
        logging.exception("Chat failed.")
        return error_response("The assistant could not answer. Please try again.")


@app.route("/chat/status")
@admin_required
def chat_status():
    return jsonify({"model": chatbot.MODEL_ID, "state": chatbot.model_status()})

# Enable Security Review in System Analytics.
from security_features import install_security_features
log_attendance_event = install_security_features(app, get_db, admin_required, recognition)

from admin_accounts import install_admin_accounts
authenticate_admin_account = install_admin_accounts(app, get_db, admin_required)
