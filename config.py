import os

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-before-production")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATABASE}")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHAT_MEMORY_LIMIT = int(os.environ.get("CHAT_MEMORY_LIMIT", "12"))

USE_CONSOLE_EMAIL = os.environ.get("USE_CONSOLE_EMAIL", "true").lower() == "true"

MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "email@gmail.com")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "your_app_password")
MAIL_USE_TLS = True

MODEL_PATH = "models/diabetes_model.pkl"
SCALER_PATH = "models/scaler.pkl"
