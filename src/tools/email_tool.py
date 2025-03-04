from typing import Type, Optional
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
    image_url: Optional[str] = Field(description="Optional image URL to include in the email")

class EmailTool(BaseTool):
    description: str = """Sends emails using the provided information.
    Input should be in the format: email(recipient, subject, body, image_url - optional)
    
    - recipient: The email address of the recipient
    - subject: The subject line of the email
    - body: The main content of the email
    - image_url: Optional URL of an image to include in the email
    
    Example: email("user@example.com", "Daily Bitcoin Report", "Here's your daily Bitcoin price report...")
    
    This tool will only send emails and confirm their status. It will not generate content or modify the provided email details.
    """
    name: str = "email_sender"
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
        image_url: Optional[str] = None
    ) -> str:
        """Send an email"""
        try:
            # Validate email format
            if '@' not in to:
                return "Invalid email address format"

            # Create message
            msg = MIMEMultipart("alternative")
            msg['From'] = self._sender_email
            msg['To'] = to
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))

            if image_url:
                html_body = f"""
                <html>
                    <body>
                        <p>{body}</p>
                        <img src="{image_url}" alt="Image">
                    </body>
                </html>
                """
                html_part = MIMEText(html_body, "html")
                msg.attach(html_part)

            with self._get_smtp_connection() as smtp:
                smtp.send_message(msg)

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

