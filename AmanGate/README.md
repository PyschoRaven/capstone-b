# AmanGate — Mohammed's latest version

Face-recognition gate access for the university: gate terminal, admin portal and analytics.

## Run it

```
pip install -r requirements.txt
python main.py
```

Then open:

- Gate terminal: http://127.0.0.1:8000/
- Admin portal: http://127.0.0.1:8000/admin (first login: `admin` / `admin1234`)

The first start creates `data/amangate.db`, which is empty. Add users from **Add User**.

## Chatbot model (optional)

The model is 1.2 GB, too big for GitHub. Download it into the `models/` folder:

https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf

Without it, the chatbot still answers its built-in questions.

## Before a real deployment

Set these environment variables. The defaults in the code are public:

- `ADMIN_PASSWORD`: the main admin's first password
- `SECRET_KEY`: protects login sessions
- `EMBED_KEY`: encrypts stored faces. Keep it safe: if it changes, everyone must be re-enrolled.

## What's new in this version

- **Dashboard:** campus picture, five cards, gate tiles, recent activity with face photos, today's entries by hour, active alerts. It zooms to fit the screen.
- **Every scan has a story:** each scan's face photo and the reason it was granted or denied. The main admin can mark denied scans as reviewed.
- **Campus Map:** live entries for each gate.
- **Access hours for each group** (students, faculty, security, visitors), set under Lock Gates.
- **Admin accounts:** one main admin and up to two view-only admins, with an audit log.
- **Account menu** on the round "A" button.
- **Voice guidance** on the gate terminal (remembered on each device) and on Add User.

`data/`, the model and log files are kept out of Git by `.gitignore`.
