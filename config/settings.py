import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3:1b")
    try:
        MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "200"))  # MB
    except ValueError:
        MAX_FILE_SIZE = 200
    SUPPORTED_FORMATS = [".csv"]
