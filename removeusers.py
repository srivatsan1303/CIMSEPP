import database

def list_users():
    """List all users in the database."""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users")
    users = cursor.fetchall()
    conn.close()

    if users:
        print("Users in the database:")
        for user in users:
            print(user['username'])
    else:
        print("No users found in the database.")

def remove_user(username):
    """Remove a specified user from the database."""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    if cursor.rowcount > 0:
        print(f"User '{username}' has been removed.")
    else:
        print(f"User '{username}' not found.")
    conn.commit()
    conn.close()

def main():
    list_users()
    username = input("Enter the username of the user to be removed: ")
    remove_user(username)

if __name__ == "__main__":
    main()
