"""
Feedback Service Module

This module handles user feedback submission, email sending, and local storage.
Supports multiple recipient emails and graceful error handling.
"""

import os
import json
import smtplib
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from besser.utilities.web_modeling_editor.backend.models import FeedbackSubmission


def submit_feedback(feedback: FeedbackSubmission) -> dict:
    """
    Process user feedback and send it via email.
    
    Sends feedback to configured email recipients and stores locally as backup.
    Supports multiple recipient emails separated by comma in FEEDBACK_EMAIL env var.
    
    Args:
        feedback: FeedbackSubmission model containing user feedback data
        
    Returns:
        dict: Status message confirming feedback receipt
        
    Environment Variables:
        FEEDBACK_EMAIL: Email addresses to receive feedback (comma-separated, required)
        SMTP_HOST: SMTP server hostname (default: smtp.gmail.com)
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USERNAME: SMTP authentication username (defaults to first email)
        SMTP_PASSWORD: SMTP authentication password or app password
        
    Raises:
        Exception: If critical feedback storage fails
    """
    try:
        # Get email configuration from environment variables
        # Support multiple emails separated by comma (e.g., "email1@example.com,email2@example.com")
        feedback_emails_str = os.getenv("FEEDBACK_EMAIL", "").strip()
        feedback_emails = [email.strip() for email in feedback_emails_str.split(",") if email.strip()]
        
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME", feedback_emails[0] if feedback_emails else "")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not feedback_emails:
            print("Warning: FEEDBACK_EMAIL not configured. Feedback will only be stored locally.")
        
        # Send email if configured
        if feedback_emails and smtp_password:
            _send_feedback_email(
                feedback=feedback,
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_username=smtp_username,
                smtp_password=smtp_password,
                recipient_emails=feedback_emails
            )
        
        # Store locally (always, as backup or primary)
        _store_feedback_locally(feedback)
        
        return {
            "status": "success",
            "message": "Feedback received. Thank you for helping us improve BESSER!"
        }
    
    except Exception as e:
        print(f"Failed to process feedback: {str(e)}")
        raise Exception(f"Failed to process feedback: {str(e)}")


def _send_feedback_email(
    feedback: FeedbackSubmission,
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    recipient_emails: list
) -> None:
    """
    Send feedback email to configured recipients.
    
    Args:
        feedback: Feedback submission data
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        smtp_username: SMTP username for authentication
        smtp_password: SMTP password for authentication
        recipient_emails: List of recipient email addresses
    """
    try:
        satisfaction_emoji = {
            "happy": "😊",
            "neutral": "😐",
            "sad": "😞"
        }.get(feedback.satisfaction, "❓")
        
        # Create MIME message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"[BESSER Feedback] {satisfaction_emoji} {feedback.category or 'General'} - "
            f"{feedback.satisfaction.upper()}"
        )
        msg["From"] = smtp_username
        msg["To"] = ", ".join(recipient_emails)
        
        # Create HTML email body
        html = _create_html_email_body(feedback, satisfaction_emoji)
        
        # Create plain text version as fallback
        text = _create_text_email_body(feedback, satisfaction_emoji)
        
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
        
        # Send email with longer timeout for Gmail
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            # Send to all configured email addresses
            server.send_message(msg, to_addrs=recipient_emails)
        
        print(
            f"Feedback email sent to {', '.join(recipient_emails)}: "
            f"{feedback.satisfaction} - {feedback.category}"
        )
        
    except Exception as email_error:
        print(f"Warning: Failed to send feedback email: {str(email_error)}")
        print(f"  Host: {smtp_host}:{smtp_port}, Recipients: {', '.join(recipient_emails)}")
        # Continue anyway - we'll store locally


def _store_feedback_locally(feedback: FeedbackSubmission) -> None:
    """
    Store feedback locally in JSONL format.
    
    Args:
        feedback: Feedback submission data
        
    Raises:
        Exception: If local storage fails
    """
    try:
        feedback_dir = Path("feedback_data")
        feedback_dir.mkdir(exist_ok=True)
        feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback.dict(), ensure_ascii=False) + "\n")
        
        print(f"Feedback stored locally: {feedback.satisfaction} - {feedback.category}")
        
    except Exception as e:
        print(f"Error storing feedback locally: {str(e)}")
        raise


def _create_html_email_body(feedback: FeedbackSubmission, satisfaction_emoji: str) -> str:
    """
    Create HTML email body for feedback.
    
    Args:
        feedback: Feedback submission data
        satisfaction_emoji: Emoji representing satisfaction level
        
    Returns:
        str: HTML email body
    """
    satisfaction_colors = {
        "happy": ("#d4edda", "#155724"),
        "neutral": ("#fff3cd", "#856404"),
        "sad": ("#f8d7da", "#721c24"),
    }
    
    bg_color, text_color = satisfaction_colors.get(
        feedback.satisfaction,
        ("#f0f0f0", "#333")
    )
    
    email_row = (
        f'<tr><td style="padding: 10px; font-weight: bold;">Email:</td>'
        f'<td style="padding: 10px;"><a href="mailto:{feedback.email}">{feedback.email}</a></td></tr>'
        if feedback.email
        else ""
    )
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px; background: #f9f9f9;">
            <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                New Feedback Received
            </h2>
            
            <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; font-weight: bold; width: 150px;">Satisfaction:</td>
                        <td style="padding: 10px;">
                            <span style="background: {bg_color}; 
                                         color: {text_color}; 
                                         padding: 5px 15px; border-radius: 20px;">
                                {satisfaction_emoji} {feedback.satisfaction.upper()}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold;">Category:</td>
                        <td style="padding: 10px;">{feedback.category or 'Not specified'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold;">Timestamp:</td>
                        <td style="padding: 10px;">{feedback.timestamp}</td>
                    </tr>
                    {email_row}
                </table>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #2c3e50; margin-top: 0;">Feedback Message:</h3>
                <p style="white-space: pre-wrap;">{feedback.feedback}</p>
            </div>
            
            <div style="background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 20px 0; font-size: 12px; color: #7f8c8d;">
                <strong>User Agent:</strong><br>
                {feedback.user_agent}
            </div>
        </div>
    </body>
    </html>
    """
    return html


def _create_text_email_body(feedback: FeedbackSubmission, satisfaction_emoji: str) -> str:
    """
    Create plain text email body for feedback.
    
    Args:
        feedback: Feedback submission data
        satisfaction_emoji: Emoji representing satisfaction level
        
    Returns:
        str: Plain text email body
    """
    email_line = f"Email: {feedback.email}" if feedback.email else ""
    
    text = f"""
New BESSER Feedback Received
=============================

Satisfaction: {satisfaction_emoji} {feedback.satisfaction.upper()}
Category: {feedback.category or 'Not specified'}
Timestamp: {feedback.timestamp}
{email_line}

Feedback Message:
{feedback.feedback}

User Agent: {feedback.user_agent}
    """
    return text
