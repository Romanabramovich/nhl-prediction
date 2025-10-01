# database.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_database_engine():
    # create SQLAlchemy engine for PostgreSQL connection
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise ValueError("value error: DB_URL environment variable not set")
    
    print("Connecting with DB_URL =", os.getenv("DB_URL"))
    return create_engine(db_url, echo=False, pool_pre_ping=True)

def test_connection():
    # test database connection
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            print("database connection successful!")
            return True
    except Exception as e:
        print(f"database connection failed: {e}")
        return False
