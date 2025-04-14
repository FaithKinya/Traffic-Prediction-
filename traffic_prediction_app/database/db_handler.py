import sqlite3

class DatabaseHandler:
    def __init__(self, db_path="data/user_data.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS traffic_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER,
            weather TEXT,
            light TEXT,
            day TEXT,
            prediction TEXT
        )
        '''
        self.conn.execute(query)
        self.conn.commit()

    def insert_input(self, hour, weather, light, day, prediction):
        query = '''INSERT INTO traffic_inputs (hour, weather, light, day, prediction)
                   VALUES (?, ?, ?, ?, ?)'''
        self.conn.execute(query, (hour, weather, light, day, prediction))
        self.conn.commit()
