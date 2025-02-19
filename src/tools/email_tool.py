from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailInput(BaseModel):
    """Schema for email input"""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body content")

class EmailTool(BaseTool):
    """Tool for sending emails using SMTP"""
    name: str = "email_sender"
    description: str = """Send emails using custom SMTP server.
    Requires:
    - Recipient email address
    - Subject
    - Body content
    
    Example:
    Send an email to john@example.com with subject "Meeting Notes" and content "Here are the meeting notes..."
    """
    args_schema: Type[BaseModel] = EmailInput
    
    def __init__(self):
        super().__init__()
        self._smtp = None
        self._setup_email()
        
        os.makedirs("email_logs", exist_ok=True)

    def _setup_email(self):
        """Initialize SMTP client"""
        try:
            self._smtp_host = os.getenv('SMTP_HOST', 'mail.privateemail.com')
            self._smtp_port = int(os.getenv('SMTP_PORT', '587'))
            self._smtp_user = os.getenv('SMTP_USER')
            self._smtp_password = os.getenv('SMTP_PASSWORD')
            self._sender_email = os.getenv('SENDER_EMAIL')
            
            if not all([self._smtp_host, self._smtp_port, self._smtp_user, self._smtp_password, self._sender_email]):
                raise ValueError("SMTP credentials not found in environment variables")
                
            # Test connection
            self._get_smtp_connection()
            logger.info("SMTP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SMTP client: {e}")
            raise

    def _get_smtp_connection(self):
        """Return a persistent SMTP connection if not already created"""
        self._smtp = smtplib.SMTP_SSL(self._smtp_host, self._smtp_port)
        self._smtp.login(self._smtp_user, self._smtp_password)
        return self._smtp

    def _run(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> str:
        """Send an email"""
        try:
            # Validate email format
            if '@' not in to:
                return "Invalid email address format"

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self._sender_email
            msg['To'] = to
            msg['Subject'] = subject

            # Add body
            msg.attach(MIMEText(body, 'plain'))

            # Send email using a fresh connection
            smtp = self._get_smtp_connection()
            smtp.send_message(msg)
            smtp.quit()

            # Log the email
            self._log_email(to, subject)
            
            return f"Email sent successfully to {to}"
            
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _log_email(self, recipient: str, subject: str):
        """Log email sending operation"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "recipient": recipient,
            "subject": subject,
            "sender": self._sender_email,
            "status": "sent"
        }
        
        with open("email_logs/email_logs.jsonl", "a") as f:
            f.write(f"{str(log_entry)}\n")

