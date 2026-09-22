import pandas as pd
import sqlite3
import os

excel_file = 'Client Data Set for Student Project Workbook.xlsx'
db_path = os.path.join('database', 'churn_platform.db')

# Ensure database directory exists
os.makedirs('database', exist_ok=True)

# Read Excel File
df = pd.read_excel(excel_file, sheet_name='Customer Intelligence Data')

# Connect to SQLite DB and save dataframe
conn = sqlite3.connect(db_path)
df.to_sql('customer_churn', conn, if_exists='replace', index=False)
conn.close()

print(f"Success! {len(df)} records with all {len(df.columns)} columns imported into 'customer_churn' table.")