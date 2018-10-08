import  nibabel as nib

connection = sqlite3.connect("mydb.db")
cursor=connection.cursor()
cursor.execute('create table customers(id integer primary key ,name text)')
connection.commit()
connection.close()
