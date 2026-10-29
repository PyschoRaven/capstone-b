"""Separate admin accounts; face-enrollment users are not changed."""
from contextlib import closing
from datetime import datetime
import re
import secrets
import sqlite3
from flask import jsonify, render_template, request, session
from werkzeug.security import generate_password_hash, check_password_hash


def install_admin_accounts(app, get_db, admin_required):
    def ensure_accounts():
        with closing(get_db()) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS admin_accounts (
                id INTEGER PRIMARY KEY, username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.commit()
    app.before_request(ensure_accounts)

    def authenticate(username, password):
        with closing(get_db()) as conn:
            row = conn.execute("SELECT username,password_hash FROM admin_accounts WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        return bool(row and check_password_hash(row['password_hash'], password))

    @app.get('/admin/accounts/new')
    @admin_required
    def new_admin_account():
        session.setdefault('admin_account_csrf', secrets.token_urlsafe(32))
        return render_template('admin-accounts.html', csrf_token=session['admin_account_csrf'])

    @app.post('/admin/accounts')
    @admin_required
    def create_admin_account():
        expected = session.get('admin_account_csrf')
        supplied = request.headers.get('X-AmanGate-CSRF', '')
        if not expected or not secrets.compare_digest(expected, supplied):
            return jsonify(status='error', message='Refresh this page and try again.'), 403
        data = request.get_json(silent=True) or {}
        username, password = data.get('username'), data.get('password')
        if not isinstance(username, str) or not isinstance(password, str):
            return jsonify(status='error', message='Enter a username and password.'), 400
        username = username.strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}', username) or username.lower() == 'admin':
            return jsonify(status='error', message='Use 3–40 letters, numbers, dots, underscores or hyphens. The name admin is reserved.'), 400
        if not 8 <= len(password) <= 128:
            return jsonify(status='error', message='Use a password with 8–128 characters.'), 400
        try:
            with closing(get_db()) as conn:
                conn.execute('INSERT INTO admin_accounts (username,password_hash,created_at) VALUES (?,?,?)',
                    (username, generate_password_hash(password), datetime.now().isoformat(timespec='seconds')))
                conn.commit()
        except sqlite3.IntegrityError:
            return jsonify(status='error', message='That username already exists.'), 409
        return jsonify(status='success', message='Admin account created. They can now sign in with their username and password.'), 201

    return authenticate
