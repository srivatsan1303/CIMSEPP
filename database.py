import sqlite3
import bcrypt

DATABASE_NAME = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def setup():
    """Create the users table in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def check_credentials(username, provided_password):
    """Check if the provided credentials match those in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    stored_password = cursor.fetchone()
    conn.close()

    if stored_password is None:
        return False
    
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password['password'])

if __name__ == '__main__':
    setup()  # Set up the database when this script is run directly
