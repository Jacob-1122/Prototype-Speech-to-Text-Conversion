import os
import sqlite3
import logging
from config import CONFIG

logger = logging.getLogger(__name__)

def initialize_transcription_database():
    """Initialize SQLite database for storing transcriptions and corrections."""
    db_path = os.path.join(CONFIG["atc_dataset_path"], "transcriptions.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audio_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        sample_rate INTEGER,
        duration REAL,
        source TEXT,
        difficulty INTEGER,
        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id INTEGER,
        transcription TEXT,
        confidence REAL,
        is_corrected BOOLEAN DEFAULT 0,
        model_version TEXT,
        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (audio_id) REFERENCES audio_samples (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE,
        description TEXT,
        training_data_size INTEGER,
        date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"Initialized transcription database at {db_path}")
    return db_path
