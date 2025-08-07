import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Google Sheets Configuration
    GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_ID", "")
    SHEET_NAME = "Sheet1"
    CREDENTIALS_PATH = "credentials.json"
    
    # Policy Document
    POLICY_PDF_PATH = os.path.join(os.path.dirname(__file__), "New_doc.txt")
    
    # Authentication
    MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", 3))
    
    # RAG Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SEARCH_K = 3
    
    # Gemini Configuration
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    GEMINI_TEMPERATURE = 0.3
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    
    # Leave Application
    MIN_LEAVE_DAYS = 0.5
    MAX_LEAVE_DAYS = 30
    
    # Contact Information
    HR_EMAIL = "hr@bankname.com"
    
    # Web App Configuration
    APP_TITLE = "Bank Policy Assistant"
    APP_ICON = "🏦"
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration values are set"""
        required_fields = ["GOOGLE_SHEETS_ID", "GEMINI_API_KEY"]
        missing_fields = []
        
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return True