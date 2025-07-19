# utils/notify.py

def notify_slack(message: str):
    print(f"[Slack] {message}")

def notify_email(subject: str, body: str):
    print(f"[Email] Subject: {subject}\nBody: {body}")