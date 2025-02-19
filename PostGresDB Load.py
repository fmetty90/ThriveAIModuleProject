#!/usr/bin/env python
# coding: utf-8

# # Loading Data to PostGres

# In[1]:


import psycopg2
import csv
from psycopg2 import sql
from psycopg2.extras import execute_values

def load_csv_to_postgres(csv_file_path, schema_name, table_name, db_name, user, password, host, port):
    cursor = None
    connection = None
    try:
        # Connect to your postgres DB
        connection = psycopg2.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        cursor = connection.cursor()

        # Open the CSV file
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row

            # Create table if it does not exist
            create_table_query = sql.SQL('''
            CREATE TABLE IF NOT EXISTS {}.{} (
                {}
            )
            ''').format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(', ').join([sql.SQL("{} VARCHAR").format(sql.Identifier(col)) for col in header])
            )
            cursor.execute(create_table_query)

            # Insert each row from the CSV file
            rows = [tuple(row) for row in reader]
            insert_query = sql.SQL('INSERT INTO {}.{} ({}) VALUES %s').format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(', ').join([sql.Identifier(col) for col in header])
            )
            execute_values(cursor, insert_query, rows)

        # Commit the transaction
        connection.commit()
        
    except Exception as error:
        print(f"Error: {error}")
    
    finally:
        # Close the cursor and connection
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

# Example usage
load_csv_to_postgres(
    #csv_file_path='C:/Users/fmett/OneDrive/DATA/CONSULTING/ThriveAi/Titanic DataSet/Titanic.csv',
    csv_file_path='C:/Users/fmett/OneDrive/DATA/CONSULTING/ThriveAi/Chronic_Disease_Indicators/U.S._Chronic_Disease_Indicators_xlsx_VII.xlsx',
    schema_name='public',
    table_name='dt_chronic_disease_indicators',
    db_name='postgres',
    user='postgres',
    password='admin',
    host='localhost',
    port='5432'
)


# In[ ]:




