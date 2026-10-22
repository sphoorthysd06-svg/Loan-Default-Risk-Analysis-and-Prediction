import pymysql

print("PyMySQL imported successfully.")

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="system",
    database="loan_default_db"
)

print("Connected to MySQL successfully!")

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM loan_data")

result = cursor.fetchone()

print("Total records in MySQL:", result[0])

cursor.close()
conn.close()

print("MySQL connection closed.")