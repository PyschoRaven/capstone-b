"""AmanGate: scan explanations and a reviewable security timeline."""
from contextlib import closing
from datetime import datetime
import threading
from flask import jsonify, request, render_template

REASONS = {
    "GRANTED": "Face matched an enrolled user and this gate's access rule allowed entry.",
    "DENIED": "No enrolled face matched within the configured distance threshold.",
    "DENIED_GATE": "Face matched an enrolled user, but this gate is outside their permitted gates.",
    "DENIED_SPOOF": "The anti-spoofing model classified the captured face as a possible screen or photo.",
}


def install_security_features(app, get_db, admin_required, recognition):
    # Additive migration: existing users and scans are retained.
    schema_lock = threading.Lock()
    schema_ready = False

    def ensure_schema():
        nonlocal schema_ready
        if schema_ready:
            return
        with schema_lock:
            if schema_ready:
                return
            with closing(get_db()) as conn:
                columns = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
                if not columns:
                    conn.execute("""CREATE TABLE scans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT NOT NULL,
                        user_id TEXT, name TEXT NOT NULL, confidence REAL, status TEXT NOT NULL)""")
                    columns = {"id", "scanned_at", "user_id", "name", "confidence", "status"}
                for column in ("gate_id", "explanation", "spoof_check"):
                    if column not in columns:
                        conn.execute("ALTER TABLE scans ADD COLUMN " + column + " TEXT")
                conn.execute("""CREATE TABLE IF NOT EXISTS scan_reviews (
                    scan_id INTEGER PRIMARY KEY, reviewed_at TEXT NOT NULL)""")
                conn.commit()
            schema_ready = True


    app.before_request(ensure_schema)

    def log_scan(user_id, name, confidence_score, status):
        data = request.get_json(silent=True) or {}
        gate = data.get("gate")
        gate = gate if gate in {"main", "male", "female", "vehicle"} else "main"
        # scan_face records the actual spoof-check outcome in Flask request state.
        from flask import g
        check = getattr(g, "security_spoof_check", "unknown")
        with closing(get_db()) as conn:
            conn.execute("""INSERT INTO scans
                (scanned_at, user_id, name, confidence, status, gate_id, explanation, spoof_check)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, name,
                 confidence_score, status, gate, REASONS.get(status, "Recorded scan result."), check))
            conn.commit()

    @app.after_request
    def explain_scan(response):
        if request.path == "/scan" and response.is_json:
            payload = response.get_json()
            if isinstance(payload, dict):
                explanations = {
                    "success": REASONS["GRANTED"], "denied": REASONS["DENIED"],
                    "not_permitted": REASONS["DENIED_GATE"], "spoof": REASONS["DENIED_SPOOF"],
                    "lockdown": "Emergency lockdown is active; recognition was not performed.",
                    "gate_locked": "This gate is closed; recognition was not performed.",
                    "multiple_faces": "More than one sufficiently large face was detected.",
                    "no_face": "No usable face embedding was obtained from the supplied frames.",
                }
                payload["explanation"] = explanations.get(payload.get("status"), payload.get("message", "No access decision recorded."))
                response.set_data(app.json.dumps(payload))
        return response

    @app.route("/security-review")
    @admin_required
    def security_review_page():
        return render_template("security_review.html")

    @app.route("/security-review/events")
    @admin_required
    def security_review_events():
        # Bound the timeline; newest 200 records, including successful scans.
        with closing(get_db()) as conn:
            rows = conn.execute("""SELECT s.id, s.scanned_at, s.user_id, s.name,
                s.confidence, s.status, s.gate_id, s.explanation, s.spoof_check, r.reviewed_at
                FROM scans s LEFT JOIN scan_reviews r ON r.scan_id = s.id
                ORDER BY s.scanned_at DESC, s.id DESC LIMIT 200""").fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["explanation"] = event["explanation"] or REASONS.get(event["status"], "Legacy scan result.")
            events.append(event)
        return jsonify(events=events)

    @app.route("/security-review/events/<int:scan_id>/review", methods=["POST"])
    @admin_required
    def review_security_event(scan_id):
        # Only the same-origin review page may change review state.
        if request.headers.get("X-AmanGate-Review") != "1":
            return jsonify(message="Missing review header"), 403
        with closing(get_db()) as conn:
            row = conn.execute("SELECT status FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return jsonify(message="Scan not found"), 404
            if not row["status"].startswith("DENIED"):
                return jsonify(message="Only denied scans need review"), 400
            conn.execute("INSERT OR IGNORE INTO scan_reviews (scan_id, reviewed_at) VALUES (?, ?)",
                         (scan_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        return jsonify(status="success")

    return log_scan
