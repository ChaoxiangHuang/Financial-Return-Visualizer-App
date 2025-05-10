import sqlite3

# 1) Connect (this creates test.db if it doesn't exist)
conn = sqlite3.connect("test.db")
cur  = conn.cursor()

# 2) Make a table
cur.execute("""
  CREATE TABLE IF NOT EXISTS returns (
    Date   TEXT PRIMARY KEY,
    Return REAL
  );
""")

# 3) Insert some sample rows
sample = [
  ("2025-01-02",  0.0012),
  ("2025-01-03", -0.0005),
  ("2025-01-06",  0.0023),
]
cur.executemany("INSERT OR REPLACE INTO returns VALUES (?, ?);", sample)

conn.commit()
conn.close()
