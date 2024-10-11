import sqlite3
import csv

def export_db_to_csv(db_file, csv_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_interactions")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)
    conn.close()
    print(f"Data exported to {csv_file} successfully.")

db_file = 'cats.db'
csv_file = 'cats.csv'
export_db_to_csv(db_file, csv_file)
