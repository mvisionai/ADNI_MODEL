import  nibabel as nib
import  argparse
import sqlite3

connection = sqlite3.connect("mydb.db")
cursor=connection.cursor()
cursor.execute('create table customers(id integer primary key ,name text)')
connection.commit()
connection.close()



