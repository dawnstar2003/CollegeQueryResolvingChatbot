import mysql.connector

class Database:
    def __init__(self, host, user, password, database):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor()

    def fetch_intents_data(self):
        self.cursor.execute("SELECT tag, patterns, responses FROM intents")
        return self.cursor.fetchall()

