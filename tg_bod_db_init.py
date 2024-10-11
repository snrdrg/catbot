import sqlite3

conn = sqlite3.connect('cats.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT,
        predicted_class INTEGER,
        ground_truth INTEGER
    )
''')
conn.commit()
conn.close()
