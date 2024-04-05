import database
import sqlite3

def add_user(username, password):
    """Add a new user to the database."""
    hashed_password = database.hash_password(password)
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
    finally:
        conn.close()

# Example usage - replace 'username' and 'password' with actual values
add_user('jyoung@njit.edu', 'a1b2c3d4')
add_user('sj796@njit.edu', '118210ZJLJ7dw@')
#add_user('user2', 'password2')
# ... Add more users as needed
