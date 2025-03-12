import sqlite3

class DatabaseManager:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()
        self._init_database()

    def _init_database(self):
        # Create the Employee table
        self.cursor.execute('''
        CREATE TABLE Employee (
            employee_name TEXT NOT NULL,
            employee_email TEXT UNIQUE NOT NULL,
            org_name TEXT,
            designation TEXT NOT NULL,
            years_of_experience REAL NOT NULL,
            salary REAL,
            location TEXT,
            hire_date TEXT
        );
        ''')
        
        # Sample data
        self.employees = [
            ("Alice Johnson", "alice.johnson@example.com", "TechCorp", "Software Engineer", 3.5, 70000, "New York", "2019-06-15"),
            ("Bob Smith", "bob.smith@example.com", "TechCorp", "Product Manager", 7.0, 95000, "San Francisco", "2016-02-10"),
            # ...existing employee data...
        ]
        
        # Insert data
        self.cursor.executemany('''
        INSERT INTO Employee (employee_name, employee_email, org_name, designation, years_of_experience, salary, location, hire_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', self.employees)
        self.connection.commit()

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if not rows:
                return "No results found"
            
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def close(self):
        self.connection.close()