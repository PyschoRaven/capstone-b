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
import secrets
import sqlite3
import threading
import time

import cv2
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
# Lax: the browser does not send the login cookie with requests started by other
# websites, so another site cannot make a logged-in admin's browser lock the gates.
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

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

# Admin accounts (admins table):
#   "main"   — full control; manages the other accounts on the Accounts tab
#   "viewer" — view-only: sees every page, cannot change anything
# The main account is created on first start as "admin" with ADMIN_PASSWORD;
# after that its password is changed in the portal. Every change is written to
# the audit log with the account that made it.
MAIN_ADMIN_USERNAME = "admin"
MAX_VIEWERS         = 2
USERNAME_RE         = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
MIN_PASSWORD_LEN    = 8
# Checked when a username does not exist, so a wrong username takes as long as a wrong password
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def current_admin():
    """
    The logged-in admin account, or None. Read from the database on every
    request, so removing an account or changing its password logs it out at once.
    """
    admin_id = session.get("admin_id")
    if admin_id is None:
        return None
    with get_db() as conn:
        row = conn.execute("SELECT id, username, role, session_token FROM admins WHERE id = ?",
                           (admin_id,)).fetchone()
    # The token is renewed on every password change, which ends the older sessions
    if not row or not secrets.compare_digest(str(session.get("admin_key") or ""), row["session_token"] or "x"):
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def _require(role=None):
    def wrap(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            account = current_admin()
            if not account:
                for key in ("admin_id", "admin_key", "chat_last"):
                    session.pop(key, None)
                return jsonify({"status": "error", "message": "Unauthorized"}), 401
            if role and account["role"] != role:
                return jsonify({"status": "error",
                                "message": "View-only account - only the main admin can do this."}), 403
            g.admin = account
            return f(*args, **kwargs)
        return decorated
    return wrap


admin_required      = _require()          # any admin account, including view-only
main_admin_required = _require("main")    # changes: only the main admin


def audit(action, detail="", username=None):
    """Write one line to the audit log: who did what, and when."""
    try:
        who = username or getattr(g, "admin", {}).get("username") or "-"
        with get_db() as conn:
            conn.execute("INSERT INTO audit_log (at, username, action, detail) VALUES (?, ?, ?, ?)",
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), who, action, detail))
            conn.commit()
    except Exception:
        logging.exception("Could not write the audit log.")


@app.route("/admin/login", methods=["POST"])
@limiter.limit("5 per minute")
def admin_login():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    with get_db() as conn:
        row = conn.execute("SELECT id, username, role, password_hash, session_token FROM admins WHERE username = ?",
                           (username,)).fetchone()
    if not check_password_hash(row["password_hash"] if row else _DUMMY_HASH, password) or not row:
        audit("Login failed", f"username: {username[:40] or '(empty)'}", username="-")
        return jsonify({"status": "error", "message": "Incorrect username or password. Please try again."}), 401
    session.clear()
    session["admin_id"]  = row["id"]
    session["admin_key"] = row["session_token"]
    with get_db() as conn:
        conn.execute("UPDATE admins SET last_login = ? WHERE id = ?",
                     (datetime.now().isoformat(timespec="seconds"), row["id"]))
        conn.commit()
    audit("Logged in", "", username=row["username"])
    return jsonify({"status": "success", "username": row["username"], "role": row["role"]})


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    account = current_admin()
    if account:
        audit("Logged out", "", username=account["username"])
    for key in ("admin_id", "admin_key", "chat_last"):
        session.pop(key, None)
    return jsonify({"status": "success"})


@app.route("/admin/session")
def admin_session():
    account = current_admin()
    if not account:
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "username": account["username"], "role": account["role"]})


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
        # Admin accounts and the audit log of what each account did
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL,
                session_token TEXT,
                created_at    TEXT,
                last_login    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                at       TEXT NOT NULL,
                username TEXT,
                action   TEXT NOT NULL,
                detail   TEXT
            )
        """)
        # Denied scans an admin has looked at (Scan History -> Mark reviewed)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_reviews (
                scan_id     INTEGER PRIMARY KEY,
                reviewed_at TEXT NOT NULL,
                reviewed_by TEXT
            )
        """)
        if not conn.execute("SELECT 1 FROM admins WHERE role = 'main'").fetchone():
            conn.execute(
                "INSERT INTO admins (username, password_hash, role, session_token, created_at) VALUES (?, ?, 'main', ?, ?)",
                (MAIN_ADMIN_USERNAME, generate_password_hash(ADMIN_PASSWORD), secrets.token_hex(16),
                 datetime.now().isoformat(timespec="seconds")))
            logging.info("Created the main admin account '%s'.", MAIN_ADMIN_USERNAME)
        # Default access hours, set once (the main admin can change them later)
        if not conn.execute("SELECT 1 FROM settings WHERE key = 'hours_seeded'").fetchone():
            for role, hours in DEFAULT_ACCESS_HOURS.items():
                set_setting(conn, f"hours:{role}", hours)
            set_setting(conn, "hours_seeded", "1")
        # Every scan, linked to the user's ID (NULL for unknown faces and spoof attempts)
        conn.execute(SCANS_TABLE)
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

SCANS_TABLE = """
    CREATE TABLE IF NOT EXISTS scans (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        scanned_at TEXT NOT NULL,
        user_id    TEXT,
        name       TEXT NOT NULL,
        confidence REAL,
        status     TEXT NOT NULL,
        gate       TEXT,
        spoof_check  TEXT,   -- real-face check: passed / flagged / not run
        role_at_scan TEXT,   -- the person's role when scanned (for the scan's story)
        photo        BLOB    -- small JPEG of the face (Scan History detail)
    )
"""
SCAN_EXTRA_COLUMNS = {"spoof_check": "TEXT", "role_at_scan": "TEXT", "photo": "BLOB"}
SCAN_PHOTO_PX      = 160   # longest side of the saved face photo


def scan_thumbnail(frame, box):
    """A small JPEG of the face (with a margin) for the scan's story, or None."""
    try:
        x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0:
            return None
        height, width = frame.shape[:2]
        pad  = int(max(w, h) * 0.2)
        crop = frame[max(0, y - pad):min(height, y + h + pad), max(0, x - pad):min(width, x + w + pad)]
        if not crop.size:
            return None
        scale = min(1.0, SCAN_PHOTO_PX / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return encoded.tobytes() if ok else None
    except Exception:
        logging.exception("Could not make the scan photo; the scan is still logged.")
        return None


def log_attendance_event(user_id, name, confidence_score, status, gate="",
                         photo=None, spoof_check=None, role=None):
    """
    Record a scan in the scans table. user_id is the enrolled user's ID, or None
    for unknown faces and spoof attempts. The name is kept as it was at scan time,
    so scans of removed or expired users stay readable. `gate` (main / male /
    female / vehicle) is the gate the scan happened at; photo, spoof_check and
    role feed the scan's story in Scan History.
    """
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scans (scanned_at, user_id, name, confidence, status, gate, spoof_check, role_at_scan, photo) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, name, confidence_score, status, gate or None,
             spoof_check, role, sqlite3.Binary(photo) if photo else None),
        )
        conn.commit()


def upgrade_scans_table(conn):
    """
    An earlier test version stored scans with a granted (0/1) column instead of a
    status. Rebuild that table once with a status column, keeping every scan.
    Tables made before the gate was recorded get an empty gate column.
    """
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
    if "granted" in columns and "status" not in columns:
        conn.execute("ALTER TABLE scans RENAME TO scans_old")
        conn.execute(SCANS_TABLE)
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
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
    if "gate" not in columns:
        conn.execute("ALTER TABLE scans ADD COLUMN gate TEXT")
    for column, kind in SCAN_EXTRA_COLUMNS.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE scans ADD COLUMN {column} {kind}")
    conn.commit()


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
                # Newer lines end with the gate (possibly empty); older ones end with the status
                gate = ""
                if parts[-1].strip() in VALID_GATES or (parts[-1] == "" and len(parts) >= 5):
                    gate = parts.pop().strip()
                try:
                    at = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                    confidence = float(parts[-2].rstrip("%"))
                except ValueError:
                    continue
                name   = ",".join(parts[1:-2])  # names were written unquoted
                status = parts[-1].strip()
                matches = ids.get(name, []) if status in ("GRANTED", "DENIED_GATE") else []
                conn.execute(
                    "INSERT INTO scans (scanned_at, user_id, name, confidence, status, gate) VALUES (?, ?, ?, ?, ?, ?)",
                    (at.strftime("%Y-%m-%d %H:%M:%S"), matches[0] if len(matches) == 1 else None,
                     name, confidence, status, gate or None),
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
            SELECT s.id, s.scanned_at, s.user_id, COALESCE(u.name, s.name) AS name, s.confidence, s.status, s.gate,
                   s.spoof_check, COALESCE(s.role_at_scan, u.role) AS role, (s.photo IS NOT NULL) AS has_photo,
                   r.reviewed_at, r.reviewed_by
            FROM scans s
            LEFT JOIN users u ON u.id = s.user_id
            LEFT JOIN scan_reviews r ON r.scan_id = s.id
            ORDER BY s.scanned_at, s.id
        """).fetchall()
    return [{
        "id":         r["id"],
        "at":         datetime.strptime(r["scanned_at"], "%Y-%m-%d %H:%M:%S"),
        "user_id":    r["user_id"],
        "name":       r["name"],
        "confidence": r["confidence"] or 0.0,
        "granted":    r["status"] == "GRANTED",
        # DENIED (unknown face), DENIED_GATE (wrong gate), DENIED_TIME (outside hours), DENIED_SPOOF
        "status":     r["status"],
        "gate":       r["gate"] or "",   # "" for scans logged before gates were recorded
        "spoof_check": r["spoof_check"],  # None for scans logged before it was recorded
        "role":        r["role"],
        "has_photo":   bool(r["has_photo"]),
        "reviewed_at": r["reviewed_at"],
        "reviewed_by": r["reviewed_by"],
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
        if e.get("status") == "DENIED_TIME":
            return "Outside access hours", f"{e['name']} came outside their allowed hours"
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

        serialize = serialize_scan
        alerts = build_scan_alerts([e for e in todays if not e["granted"]])

        # Granted entries in each hour of today (12 AM .. 11 PM) for the Entry activity chart
        today_hourly = [0] * 24
        for e in todays:
            if e["granted"]:
                today_hourly[e["at"].hour] += 1
        entrants = {e["user_id"] or e["name"] for e in todays if e["granted"]}
        visitors = {e["user_id"] or e["name"] for e in todays if e["granted"] and e["role"] == "visitor"}

        return jsonify({
            "status":    "success",
            "now":       now.isoformat(timespec="seconds"),
            "today":     {"granted": granted, "denied": len(todays) - granted, "total": len(todays),
                          "entrants": len(entrants), "visitors": len(visitors)},
            "yesterday": {"granted": sum(1 for e in events if e["granted"] and e["at"].date() == yesterday)},
            # Denied scans (any day) that no admin has reviewed yet
            "needs_review": sum(1 for e in events if not e["granted"] and not e["reviewed_at"]),
            "hourly":    hourly,
            "today_hourly": today_hourly,
            "recent":    [serialize(e) for e in reversed(events[-7:])],
            "last_scan": events[-1]["at"].isoformat(timespec="seconds") if events else None,
            "alerts":    [{**a, "at": a["at"].isoformat(timespec="seconds")} for a in alerts],
        })
    except Exception:
        logging.exception("Dashboard stats failed.")
        return error_response("Could not load dashboard data.")


def serialize_scan(e):
    return {"id": e.get("id"), "at": e["at"].isoformat(timespec="seconds"), "name": e["name"],
            "user_id": e.get("user_id"), "role": e.get("role"),
            "confidence": e["confidence"], "granted": e["granted"],
            "status": e["status"], "gate": e["gate"],
            "photo": f"/scans/{e['id']}/photo" if e.get("has_photo") else None,
            "reviewed_at": e.get("reviewed_at"), "reviewed_by": e.get("reviewed_by"),
            "story": scan_story(e)}


# --------------------------
# Each scan's story: why it was granted or denied, step by step
# --------------------------
# The steps follow the order /scan checks them in: face -> identity -> real face
# -> gate permission -> access hours. A step that was never reached is "skip".

GRANT_FROM = round((1 - THRESHOLD) * 100)   # match % needed for access (65%)


def scan_story(e):
    status, gate = e["status"], GATE_NAMES.get(e["gate"], "this gate")
    conf, who    = e["confidence"], e["name"]
    known        = status in ("GRANTED", "DENIED_GATE", "DENIED_TIME")
    spoof        = e.get("spoof_check")

    def step(title, state, text):
        return {"title": title, "state": state, "text": text}

    identity = (step("Identity match", "good", f"Matched {who} ({conf:.0f}% match; {GRANT_FROM}% needed).") if known else
                step("Identity match", "bad", f"No enrolled face matched (closest {conf:.0f}%; {GRANT_FROM}% needed).")
                if status == "DENIED" else
                step("Identity match", "skip", "Not decided - the fake-face check stopped the scan."))
    if status == "DENIED_SPOOF" or spoof == "flagged":
        real = step("Real face check", "bad", "Looked like a screen or a printed photo.")
    elif spoof == "passed":
        real = step("Real face check", "good", "A real face, not a screen or photo.")
    else:
        real = step("Real face check", "skip", "Not recorded for this scan.")
    gate_step = (step("Gate permission", "good", f"Allowed to use the {gate}.") if status in ("GRANTED", "DENIED_TIME") else
                 step("Gate permission", "bad", f"Not permitted at the {gate}.") if status == "DENIED_GATE" else
                 step("Gate permission", "skip", "Not checked."))
    hours_step = (step("Access hours", "good", "Inside the allowed hours for their role.") if status == "GRANTED" else
                  step("Access hours", "bad", "Outside the allowed hours for their role.") if status == "DENIED_TIME" else
                  step("Access hours", "skip", "Not checked."))

    summary = {
        "GRANTED":      f"{who} matched their registered face at the {gate}, is allowed there, and came inside their hours.",
        "DENIED":       "The face did not match anyone registered closely enough, so the gate stayed closed.",
        "DENIED_GATE":  f"{who} was recognised, but is not permitted to use the {gate}.",
        "DENIED_TIME":  f"{who} was recognised at an allowed gate, but outside the access hours for their role.",
        "DENIED_SPOOF": "The camera saw a screen or printed photo instead of a real face, so the scan was refused.",
    }.get(status, "Recorded scan result.")
    return {
        "title":   "Why was access granted?" if status == "GRANTED" else "Why was access denied?",
        "label":   {"GRANTED": "Access granted", "DENIED_GATE": "Wrong gate", "DENIED_TIME": "Outside hours",
                    "DENIED_SPOOF": "Fake face"}.get(status, "Unknown face"),
        "steps":   [step("Face detected", "good", "A face was found in the camera picture."),
                    identity, real, gate_step, hours_step],
        "summary": summary,
    }


# --------------------------
# Full scan history (Dashboard -> Recent scans -> View all)
# --------------------------

SCAN_PERIODS = {"today": 0, "7d": 6, "30d": 29}


@app.route("/scans")
@admin_required
def scan_history():
    """
    Every scan, newest first, with filters:
      q       — part of a name or ID ("unknown" finds unknown faces)
      gate    — main / male / female / vehicle / none (scans logged before gates were recorded)
      result  — granted / denied
      review  — pending (denied, not reviewed yet) / reviewed (denied, reviewed)
      period  — today / 7d / 30d (default: all time)
      page    — 1, 2, ... (25 per page)
    """
    try:
        args   = request.args
        q      = (args.get("q") or "").strip().lower()
        gate   = args.get("gate") or ""
        result = args.get("result") or ""
        review = args.get("review") or ""
        period = args.get("period") or ""
        events = read_attendance_log()

        if period in SCAN_PERIODS:
            since = datetime.now().date() - timedelta(days=SCAN_PERIODS[period])
            events = [e for e in events if e["at"].date() >= since]
        if gate in VALID_GATES:
            events = [e for e in events if e["gate"] == gate]
        elif gate == "none":
            events = [e for e in events if not e["gate"]]
        if q:
            events = [e for e in events if q in e["name"].lower() or q in (e["user_id"] or "")]
        denied = [e for e in events if not e["granted"]]
        counts = {"total": len(events), "granted": len(events) - len(denied), "denied": len(denied),
                  "needs_review": sum(1 for e in denied if not e["reviewed_at"]),
                  "reviewed":     sum(1 for e in denied if e["reviewed_at"])}
        if result == "granted":
            events = [e for e in events if e["granted"]]
        elif result == "denied":
            events = denied
        if review == "pending":
            events = [e for e in events if not e["granted"] and not e["reviewed_at"]]
        elif review == "reviewed":
            events = [e for e in events if not e["granted"] and e["reviewed_at"]]

        per_page = min(max(5, int(args.get("per_page") or 25)), 100)
        pages    = max(1, -(-len(events) // per_page))
        page     = min(max(1, int(args.get("page") or 1)), pages)
        newest   = list(reversed(events))
        return jsonify({
            "status": "success",
            "scans":  [serialize_scan(e) for e in newest[(page - 1) * per_page: page * per_page]],
            "total":  len(events), "page": page, "pages": pages, "per_page": per_page,
            "counts": counts,
        })
    except Exception:
        logging.exception("Scan history failed.")
        return error_response("Could not load the scan history.")


@app.route("/scans/<int:scan_id>/photo")
@admin_required
def scan_photo(scan_id):
    """The small face photo saved with a scan (private, never cached)."""
    with get_db() as conn:
        row = conn.execute("SELECT photo FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if not row or row["photo"] is None:
        return jsonify({"status": "error", "message": "No photo for this scan"}), 404
    return app.response_class(bytes(row["photo"]), mimetype="image/jpeg",
                              headers={"Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff"})


@app.route("/scans/<int:scan_id>/review", methods=["POST"])
@limiter.limit("60 per minute")
@main_admin_required
def review_scan(scan_id):
    """Mark a denied scan as reviewed. It does not change anyone's access."""
    with get_db() as conn:
        row = conn.execute("SELECT name, status, scanned_at, gate FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if not row:
            return jsonify({"status": "error", "message": "Scan not found"}), 404
        if not row["status"].startswith("DENIED"):
            return error_response("Only denied scans need a review.")
        conn.execute("INSERT OR IGNORE INTO scan_reviews (scan_id, reviewed_at, reviewed_by) VALUES (?, ?, ?)",
                     (scan_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), g.admin["username"]))
        conn.commit()
    audit("Reviewed scan", f"{row['name']} at {GATE_NAMES.get(row['gate'], 'unknown gate')}, {row['scanned_at']}")
    return jsonify({"status": "success"})


# --------------------------
# Campus map — live activity at each gate
# --------------------------
# Entry-only: the gates scan people coming IN, not going out, so the map shows
# entries per gate (today / last hour), not how many people are inside now.

@app.route("/campus-map")
@admin_required
def campus_map():
    try:
        events   = read_attendance_log()
        now      = datetime.now()
        hour_ago = now - timedelta(hours=1)
        todays   = [e for e in events if e["at"].date() == now.date()]

        gates = []
        with get_db() as conn:
            for gate_id in GATE_NAMES:
                info    = gate_info(conn, gate_id)
                mine    = [e for e in todays if e["gate"] == gate_id]
                granted = [e for e in mine if e["granted"]]
                denied  = [e for e in mine if not e["granted"]]
                last    = next((e for e in reversed(events) if e["gate"] == gate_id), None)
                info.update({
                    "entries_today": len(granted),
                    "entries_hour":  sum(1 for e in granted if e["at"] >= hour_ago),
                    "denied_today":  len(denied),
                    "spoof_today":   sum(1 for e in denied if e["status"] == "DENIED_SPOOF"),
                    "wrong_gate_today": sum(1 for e in denied if e["status"] == "DENIED_GATE"),
                    # Denied scans in the last hour -> warning marker on the map
                    "denied_hour":   [serialize_scan(e) for e in denied if e["at"] >= hour_ago][-5:],
                    "recent":        [serialize_scan(e) for e in mine[-5:]][::-1],
                    "last":          serialize_scan(last) if last else None,
                })
                gates.append(info)

        granted_today = [e for e in todays if e["granted"]]
        return jsonify({
            "status":   "success",
            "now":      now.isoformat(timespec="seconds"),
            "lockdown": lockdown_enabled,
            "gates":    gates,
            "campus": {
                "entries_today": len(granted_today),
                "entries_hour":  sum(1 for e in granted_today if e["at"] >= hour_ago),
                "denied_today":  len(todays) - len(granted_today),
                # Today's scans from before the gate was written to the log
                "no_gate_today": sum(1 for e in todays if not e["gate"]),
            },
        })
    except Exception:
        logging.exception("Campus map failed.")
        return error_response("Could not load the campus map.")


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
            hours      = access_hours(conn)
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
        last_face  = None
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
            last_face = (face["frame"], face["box"])   # photo for the scan's story if nobody matched

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
    spoof = None
    if best_frame is not None and recognition.ANTI_SPOOF_MODE != "off":
        try:
            spoof = recognition.check_spoof(best_frame, best_box)
        except Exception:
            logging.exception("Anti-spoofing check failed.")
    # For the scan's story (Scan History): the face photo and the real-face result
    photo_frame = (best_frame, best_box) if best_frame is not None else last_face
    photo       = scan_thumbnail(*photo_frame) if photo_frame else None
    spoof_check = "not run" if spoof is None else ("passed" if spoof[0] else "flagged")
    if recognition.ANTI_SPOOF_MODE == "enforce" and spoof and not spoof[0]:
        logging.info("Scan result: SPOOF detected - access denied.")
        log_attendance_event(None, "Spoof attempt", 0, "DENIED_SPOOF", gate, photo, spoof_check)
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
        # Time rule: the access hours set for this person's role (empty = any time)
        role_hours = hours.get(best_match["role"] or "student", "") if recognized else ""
        in_hours   = within_hours(role_hours, datetime.now())

        logging.info(
            "Scan result: gate=%s%s user=%s distance=%.4f confidence=%.1f%% yaw=%s spoof(real,score)=%s result=%s",
            gate, " (vehicle)" if vehicle else "",
            best_match["name"] if best_match else "Unknown",
            best_dist, best_confidence, yaws, spoof,
            ("GRANTED" if in_hours else f"DENIED (outside access hours {role_hours})") if permitted
            else ("DENIED (not permitted at this gate)" if recognized else "DENIED"),
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
            log_attendance_event(best_match["id"], best_match["name"], best_confidence, "DENIED_GATE", gate, photo, spoof_check, best_match["role"])
            return jsonify({
                "status":  "not_permitted",
                "user":    best_match["name"],
                "message": f"Not permitted at the {gate_state['name']}",
            })

        if recognized and not in_hours:
            log_attendance_event(best_match["id"], best_match["name"], best_confidence, "DENIED_TIME", gate, photo, spoof_check, best_match["role"])
            return jsonify({
                "status":  "outside_hours",
                "user":    best_match["name"],
                "hours":   hours_label(role_hours),
                "message": f"Outside access hours ({hours_label(role_hours)})",
            })

        if recognized:
            log_attendance_event(best_match["id"], best_match["name"], best_confidence, "GRANTED", gate, photo, spoof_check, best_match["role"])
            return jsonify({
                "status":     "success",
                "user":       best_match["name"],
                "role":       best_match["role"].capitalize(),
                "confidence": best_confidence,
                "box":        face_box,
            })

        log_attendance_event(None, "Unknown", best_confidence, "DENIED", gate, photo, spoof_check)
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
@main_admin_required
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
            previous = conn.execute("SELECT name FROM users WHERE id = ?", (uid,)).fetchone()
            conn.execute("""
                INSERT OR REPLACE INTO users (id, name, role, embedding, enrolled_at, gender, gates, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (uid, name, role, dump_embedding(embedding),
                  now.isoformat(timespec="seconds"), gender, gates, expires_at))
            conn.commit()
        invalidate_users_cache()
        detail = f"{name} (ID {uid}, {role}, gates: {gates or 'none'})"
        if previous:
            audit("Re-enrolled user", detail + (f", was {previous['name']}" if previous["name"] != name else ""))
        else:
            audit("Enrolled user", detail)

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
@main_admin_required
def delete_user():
    """Remove a user by their unique ID."""
    try:
        data = request.get_json(silent=True) or {}
        uid  = data.get("id", "").strip()
        if not uid:
            return error_response("No ID provided")

        with get_db() as conn:
            row    = conn.execute("SELECT name FROM users WHERE id = ?", (uid,)).fetchone()
            result = conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            conn.commit()

        if result.rowcount == 0:
            return error_response(f"ID '{uid}' not found")

        invalidate_users_cache()
        audit("Removed user", f"{row['name']} (ID {uid})")
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
@main_admin_required
def lockdown_on():
    global lockdown_enabled
    lockdown_enabled = True
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('lockdown', '1')")
        conn.execute("INSERT INTO lockdown_events (at) VALUES (?)", (datetime.now().isoformat(timespec="seconds"),))
        conn.commit()
    audit("Emergency lockdown ON", "all gates closed")
    return jsonify({"lockdown": True, "message": "Emergency Lockdown Enabled"})


@app.route("/lockdown/off", methods=["POST"])
@limiter.limit("10 per minute")
@main_admin_required
def lockdown_off():
    global lockdown_enabled
    lockdown_enabled = False
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('lockdown', '0')")
        conn.commit()
    audit("Emergency lockdown OFF", "")
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
@main_admin_required
def update_gate(gate_id):
    """Lock / unlock one gate (its walk-in / vehicle type is fixed)."""
    if gate_id not in GATE_NAMES:
        return error_response("Unknown gate"), 404
    data = request.get_json(silent=True) or {}
    with get_db() as conn:
        was_locked = gate_info(conn, gate_id)["locked"]
        if "locked" in data:
            set_setting(conn, f"gate_locked:{gate_id}", "1" if data["locked"] else "0")
        conn.commit()
        info = gate_info(conn, gate_id)
    logging.info("Gate updated: %s locked=%s type=%s", gate_id, info["locked"], info["type"])
    if info["locked"] != was_locked:
        audit("Locked gate" if info["locked"] else "Unlocked gate", info["name"])
    return jsonify({"status": "success", "gate": info})


# --------------------------
# Access hours per role (time rules)
# --------------------------
# Saved in settings as "hours:<role>" = "HH:MM-HH:MM"; empty means any time.
# A range past midnight (22:00-06:00) works too. Visitors also expire after 24 hours.

ROLE_NAMES           = {"student": "Students", "faculty": "Faculty", "security": "Security", "visitor": "Visitors"}
DEFAULT_ACCESS_HOURS = {"student": "06:00-22:00"}   # roles not listed: any time
HOURS_RE             = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


def access_hours(conn):
    return {role: get_setting(conn, f"hours:{role}", "") or "" for role in ROLE_NAMES}


def within_hours(hours, when):
    """True when `when` falls inside the "HH:MM-HH:MM" range (an empty range means any time)."""
    m = HOURS_RE.match(hours or "")
    if not m:
        return True
    start = int(m[1]) * 60 + int(m[2])
    end   = int(m[3]) * 60 + int(m[4])
    now   = when.hour * 60 + when.minute
    if start == end:
        return True
    return start <= now < end if start < end else (now >= start or now < end)


def hours_label(hours):
    return hours.replace("-", "–") if hours else "any time"


@app.route("/access-hours")
@admin_required
def get_access_hours():
    with get_db() as conn:
        return jsonify({"status": "success", "hours": access_hours(conn), "roles": ROLE_NAMES})


@app.route("/access-hours", methods=["POST"])
@limiter.limit("30 per minute")
@main_admin_required
def set_access_hours():
    data    = request.get_json(silent=True) or {}
    changes = []
    with get_db() as conn:
        current = access_hours(conn)
        for role, value in (data.get("hours") or {}).items():
            if role not in ROLE_NAMES:
                continue
            value = (value or "").strip()
            if value and not HOURS_RE.match(value):
                return error_response(f"{ROLE_NAMES[role]}: use a time range like 06:00-22:00.")
            if value and value[:5] == value[6:]:
                return error_response(f"{ROLE_NAMES[role]}: the start and end time can't be the same.")
            if value != current[role]:
                set_setting(conn, f"hours:{role}", value)
                changes.append(f"{ROLE_NAMES[role]}: {hours_label(value)}")
        conn.commit()
        result = access_hours(conn)
    if changes:
        audit("Changed access hours", "; ".join(changes))
    return jsonify({"status": "success", "hours": result})


# --------------------------
# Admin accounts + audit log (main admin only)
# --------------------------

def _account_rows(conn):
    return [dict(r) for r in conn.execute(
        "SELECT id, username, role, created_at, last_login FROM admins ORDER BY role = 'viewer', id")]


@app.route("/admins")
@main_admin_required
def list_admins():
    with get_db() as conn:
        return jsonify({"status": "success", "accounts": _account_rows(conn),
                        "max_viewers": MAX_VIEWERS, "me": g.admin["id"]})


@app.route("/admins", methods=["POST"])
@limiter.limit("20 per minute")
@main_admin_required
def add_admin():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not USERNAME_RE.match(username):
        return error_response("Username: 3-32 letters, numbers, dots, dashes or underscores.")
    if len(password) < MIN_PASSWORD_LEN:
        return error_response(f"Password: at least {MIN_PASSWORD_LEN} characters.")
    with get_db() as conn:
        if conn.execute("SELECT COUNT(*) FROM admins WHERE role = 'viewer'").fetchone()[0] >= MAX_VIEWERS:
            return error_response(f"There can be at most {MAX_VIEWERS} view-only accounts. Remove one first.")
        if conn.execute("SELECT 1 FROM admins WHERE lower(username) = lower(?)", (username,)).fetchone():
            return error_response(f"The username '{username}' is already taken.")
        conn.execute(
            "INSERT INTO admins (username, password_hash, role, session_token, created_at) VALUES (?, ?, 'viewer', ?, ?)",
            (username, generate_password_hash(password), secrets.token_hex(16), datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        accounts = _account_rows(conn)
    audit("Added view-only account", username)
    return jsonify({"status": "success", "message": f"View-only account '{username}' created.", "accounts": accounts})


@app.route("/admins/<int:account_id>/password", methods=["POST"])
@limiter.limit("10 per minute")
@main_admin_required
def set_admin_password(account_id):
    data     = request.get_json(silent=True) or {}
    password = data.get("password") or ""
    if len(password) < MIN_PASSWORD_LEN:
        return error_response(f"Password: at least {MIN_PASSWORD_LEN} characters.")
    with get_db() as conn:
        row = conn.execute("SELECT id, username, password_hash FROM admins WHERE id = ?", (account_id,)).fetchone()
        if not row:
            return error_response("Account not found."), 404
        # Your own password: confirm the current one first
        own = row["id"] == g.admin["id"]
        if own and not check_password_hash(row["password_hash"], data.get("current") or ""):
            return error_response("Your current password is not correct.")
        token = secrets.token_hex(16)   # new token: sessions using the old password end
        conn.execute("UPDATE admins SET password_hash = ?, session_token = ? WHERE id = ?",
                     (generate_password_hash(password), token, account_id))
        conn.commit()
    if own:
        session["admin_key"] = token    # stay logged in on this browser
    audit("Changed password", f"account: {row['username']}")
    return jsonify({"status": "success", "message": f"Password changed for '{row['username']}'."})


@app.route("/admins/<int:account_id>/delete", methods=["POST"])
@limiter.limit("20 per minute")
@main_admin_required
def delete_admin(account_id):
    with get_db() as conn:
        row = conn.execute("SELECT username, role FROM admins WHERE id = ?", (account_id,)).fetchone()
        if not row:
            return error_response("Account not found."), 404
        if row["role"] == "main":
            return error_response("The main admin account can't be removed.")
        conn.execute("DELETE FROM admins WHERE id = ?", (account_id,))
        conn.commit()
        accounts = _account_rows(conn)
    audit("Removed view-only account", row["username"])
    return jsonify({"status": "success", "message": f"Account '{row['username']}' removed.", "accounts": accounts})


@app.route("/audit")
@main_admin_required
def audit_log():
    """The audit log, newest first, 12 per page."""
    per_page = 12
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        pages = max(1, -(-total // per_page))
        page  = min(max(1, request.args.get("page", 1, type=int)), pages)
        rows  = conn.execute("SELECT at, username, action, detail FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
                             (per_page, (page - 1) * per_page)).fetchall()
    return jsonify({"status": "success", "entries": [dict(r) for r in rows],
                    "total": total, "page": page, "pages": pages, "per_page": per_page})


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
