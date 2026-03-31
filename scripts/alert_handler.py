import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_slack_alert(message, webhook_url):
    """
    Send alert to Slack channel.
    
    Args:
        message (str): Alert message to send
        webhook_url (str): Slack webhook URL from SLACK_WEBHOOK env var
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not webhook_url:
        print("⚠ SLACK_WEBHOOK not configured")
        return False
    
    try:
        payload = {
            "text": message,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_Alert sent at {datetime.utcnow().isoformat()}Z_"
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✓ Slack alert sent successfully")
            return True
        else:
            print(f"✗ Slack alert failed with status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"✗ Error sending Slack alert: {e}")
        return False

def send_email_alert(message, recipient):
    """
    Send alert via email.
    
    Args:
        message (str): Alert message to send
        recipient (str): Recipient email address
    
    Returns:
        bool: True if successful, False otherwise
    
    Environment Variables Required:
        - EMAIL_ADDRESS: Sender email address
        - EMAIL_PASSWORD: Email password or app token
        - SMTP_SERVER: SMTP server (default: smtp.gmail.com)
        - SMTP_PORT: SMTP port (default: 587)
    """
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    
    if not email_address or not email_password:
        print("⚠ EMAIL_ADDRESS or EMAIL_PASSWORD not configured")
        return False
    
    try:
        # Create email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "🚨 Traffic Complaint Alert"
        msg['From'] = email_address
        msg['To'] = recipient
        
        # Plain text version
        text = f"Traffic Alert\n\n{message}"
        
        # HTML version
        html = f"""\
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">🚨 Traffic Complaint Alert</h2>
                <p>{message.replace(chr(10), '<br>')}</p>
                <hr>
                <p style="font-size: 12px; color: #666;">
                    Sent at {datetime.utcnow().isoformat()}Z
                </p>
            </body>
        </html>
        """
        
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        
        print(f"✓ Email alert sent to {recipient}")
        return True
    
    except Exception as e:
        print(f"✗ Error sending email alert: {e}")
        return False

def send_twilio_sms_alert(message, phone_number):
    """
    Send alert via SMS using Twilio.
    
    Args:
        message (str): Alert message to send
        phone_number (str): Recipient phone number
    
    Returns:
        bool: True if successful, False otherwise
    
    Environment Variables Required:
        - TWILIO_ACCOUNT_SID
        - TWILIO_AUTH_TOKEN
        - TWILIO_PHONE_NUMBER (sender)
    """
    try:
        from twilio.rest import Client
    except ImportError:
        print("⚠ Twilio not installed. Run: pip install twilio")
        return False
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    sender_phone = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not account_sid or not auth_token or not sender_phone:
        print("⚠ Twilio credentials not configured")
        return False
    
    try:
        client = Client(account_sid, auth_token)
        sms = client.messages.create(
            body=message,
            from_=sender_phone,
            to=phone_number
        )
        
        print(f"✓ SMS alert sent to {phone_number} (SID: {sms.sid})")
        return True
    
    except Exception as e:
        print(f"✗ Error sending SMS alert: {e}")
        return False

def send_webhook_alert(message, webhook_url):
    """
    Send alert to custom webhook.
    
    Args:
        message (str): Alert message to send
        webhook_url (str): Webhook URL from WEBHOOK_URL env var
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not webhook_url:
        print("⚠ WEBHOOK_URL not configured")
        return False
    
    try:
        payload = {
            "alert": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code in [200, 201, 202]:
            print(f"✓ Webhook alert sent successfully")
            return True
        else:
            print(f"✗ Webhook alert failed with status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"✗ Error sending webhook alert: {e}")
        return False

if __name__ == "__main__":
    # Test the alert functions
    print("Testing Alert Handler...")
    print("=" * 60)
    
    # Test Slack (if configured)
    slack_webhook = os.getenv("SLACK_WEBHOOK")
    if slack_webhook:
        send_slack_alert("🧪 Test alert from Traffic API", slack_webhook)
    else:
        print("ℹ Slack not configured (set SLACK_WEBHOOK env var)")
    
    # Test Email (if configured)
    email_password = os.getenv("EMAIL_PASSWORD")
    if email_password:
        send_email_alert(
            "🧪 Test alert from Traffic API",
            os.getenv("ALERT_EMAIL_RECIPIENT", "admin@traffic.local")
        )
    else:
        print("ℹ Email not configured (set EMAIL_PASSWORD env var)")
    
    # Test Webhook (if configured)
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        send_webhook_alert("🧪 Test alert from Traffic API", webhook_url)
    else:
        print("ℹ Webhook not configured (set WEBHOOK_URL env var)")
    
    print("=" * 60)
