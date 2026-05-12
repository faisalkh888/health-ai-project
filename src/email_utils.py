import smtplib
from email.message import EmailMessage

from config import (
    MAIL_PASSWORD,
    MAIL_PORT,
    MAIL_SERVER,
    MAIL_USERNAME,
    MAIL_USE_TLS,
    USE_CONSOLE_EMAIL,
)


def send_email(to, subject, body, attachments=None):
    if USE_CONSOLE_EMAIL:
        print("EMAIL SENT")
        print("To:", to)
        print("Subject:", subject)
        print(body)
        for attachment in attachments or []:
            print("Attachment:", attachment["filename"])
        return True

    message = EmailMessage()
    message["From"] = MAIL_USERNAME
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    for attachment in attachments or []:
        with open(attachment["path"], "rb") as fh:
            message.add_attachment(
                fh.read(),
                maintype=attachment.get("maintype", "application"),
                subtype=attachment.get("subtype", "octet-stream"),
                filename=attachment["filename"],
            )

    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
        if MAIL_USE_TLS:
            server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(message)

    return True
