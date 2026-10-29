"""AmanGate: scan explanations and a reviewable security timeline."""
from contextlib import closing
from datetime import datetime, timedelta
import threading
import logging
from flask import jsonify, request, redirect, Response

REASONS = {
    "GRANTED": "Face matched an enrolled user and this gate's access rule allowed entry.",
    "DENIED": "No enrolled face matched within the configured distance threshold.",
    "DENIED_GATE": "Face matched an enrolled user, but this gate is outside their permitted gates.",
    "DENIED_SPOOF": "The anti-spoofing model classified the captured face as a possible screen or photo.",
}


def captured_thumbnail():
    from flask import g
    capture = getattr(g, "scan_capture", None)
    if capture is None:
        return None
    try:
        import cv2
        frame, box = capture
        x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0:
            return None
        height, width = frame.shape[:2]
        pad = int(max(w, h) * .2)
        crop = frame[max(0, y-pad):min(height, y+h+pad), max(0, x-pad):min(width, x+w+pad)]
        if not crop.size:
            return None
        scale = min(1., 160 / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return encoded.tobytes() if ok else None
    except Exception:
        logging.exception("Could not save scan thumbnail; scan logging continues.")
        return None


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
                for column in ("gate_id", "explanation", "spoof_check", "role_at_scan"):
                    if column not in columns:
                        conn.execute("ALTER TABLE scans ADD COLUMN " + column + " TEXT")
                if "photo" not in columns:
                    conn.execute("ALTER TABLE scans ADD COLUMN photo BLOB")
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
        thumbnail = captured_thumbnail()
        with closing(get_db()) as conn:
            person = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone() if user_id else None
            role_at_scan = person["role"] if person else None
            conn.execute("""INSERT INTO scans
                (scanned_at, user_id, name, confidence, status, gate_id, explanation, spoof_check, role_at_scan, photo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, name,
                 confidence_score, status, gate, REASONS.get(status, "Recorded scan result."), check, role_at_scan, thumbnail))
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
        return redirect("/system#security-review")

    @app.route("/security-review/events")
    @admin_required
    def security_review_events():
        # Bound the timeline; newest 200 records, including successful scans.
        with closing(get_db()) as conn:
            rows = conn.execute("""SELECT s.id, s.scanned_at, s.user_id, s.name,
                s.confidence, s.status, s.gate_id, s.explanation, s.spoof_check, s.role_at_scan, (s.photo IS NOT NULL) AS has_photo, r.reviewed_at
                FROM scans s LEFT JOIN scan_reviews r ON r.scan_id = s.id
                ORDER BY s.scanned_at DESC, s.id DESC LIMIT 200""").fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["explanation"] = event["explanation"] or REASONS.get(event["status"], "Legacy scan result.")
            event["photo_url"] = "/security-review/events/{}/photo".format(event["id"]) if event.pop("has_photo") else None
            events.append(event)
        return jsonify(events=events)

    @app.route("/security-review/events/<int:scan_id>/photo")
    @admin_required
    def scan_photo(scan_id):
        with closing(get_db()) as conn:
            row = conn.execute("SELECT photo FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None or row["photo"] is None:
            return jsonify(message="Photo not recorded"), 404
        return Response(bytes(row["photo"]), mimetype="image/jpeg", headers={"Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff"})

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

    @app.route("/dashboard/reference-stats")
    @admin_required
    def reference_dashboard_stats():
        now = datetime.now()
        start = now.strftime("%Y-%m-%d") + " 00:00:00"
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
        with closing(get_db()) as conn:
            rows = conn.execute("""SELECT CAST(substr(scanned_at, 12, 2) AS INTEGER) AS hour,
                COUNT(*) AS count FROM scans WHERE status = 'GRANTED'
                AND scanned_at >= ? AND scanned_at < ? GROUP BY hour""", (start, end)).fetchall()
            visitors = conn.execute("""SELECT COUNT(DISTINCT s.user_id) FROM scans s
                LEFT JOIN users u ON u.id = s.user_id
                WHERE s.status = 'GRANTED' AND s.scanned_at >= ? AND s.scanned_at < ?
                AND COALESCE(s.role_at_scan, u.role) = 'visitor'""", (start, end)).fetchone()[0]
            unique_entries = conn.execute("SELECT COUNT(DISTINCT user_id) FROM scans WHERE status = 'GRANTED' AND scanned_at >= ? AND scanned_at < ?", (start, end)).fetchone()[0]
            pending = conn.execute("""SELECT COUNT(*) FROM scans s
                LEFT JOIN scan_reviews r ON r.scan_id = s.id
                WHERE s.status LIKE 'DENIED%' AND r.scan_id IS NULL""").fetchone()[0]
        hours = {r["hour"]: r["count"] for r in rows}
        return jsonify(status="success", date=now.strftime("%Y-%m-%d"),
            hourly=[{"hour": h, "count": hours.get(h, 0)} for h in range(24)],
            visitors_today=visitors, unique_entries_today=unique_entries, needs_review=pending)

    return log_scan
