"""SQLite persistence with atomic batch writes and explicit connection lifetimes."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

DEFAULTS = {'engine': 'lexical', 'maxScore': 100, 'passPercent': 60, 'aiWeight': 70, 'useDeepseek': False}


class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS templates (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK (id=1), payload TEXT NOT NULL);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=15, factory=ClosingConnection)

    def settings(self):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM settings WHERE id=1').fetchone()
        return {**DEFAULTS, **(json.loads(row[0]) if row else {})}

    def save_settings(self, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (1, ?)', (json.dumps(value),))
        return value

    def save_records(self, values):
        with self.connect() as db:
            for value in values:
                value['id'] = uuid.uuid4().hex
                value['createdAt'] = datetime.now(timezone.utc).isoformat()
                db.execute('INSERT INTO records VALUES (?, ?, ?)',
                    (value['id'], value['createdAt'], json.dumps(value, ensure_ascii=False)))
        return values

    def records(self):
        with self.connect() as db:
            rows = db.execute('SELECT payload FROM records ORDER BY created_at DESC, id DESC').fetchall()
        return [json.loads(r[0]) for r in rows]

    def record(self, record_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM records WHERE id=?', (record_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete_record(self, record_id):
        with self.connect() as db:
            return db.execute('DELETE FROM records WHERE id=?', (record_id,)).rowcount > 0

    def templates(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM templates ORDER BY rowid DESC')]

    def save_template(self, value, template_id=None):
        value['id'] = template_id or uuid.uuid4().hex
        with self.connect() as db:
            if template_id and not db.execute('SELECT 1 FROM templates WHERE id=?', (template_id,)).fetchone():
                return None
            db.execute('INSERT OR REPLACE INTO templates VALUES (?, ?)', (value['id'], json.dumps(value, ensure_ascii=False)))
        return value

    def delete_template(self, template_id):
        with self.connect() as db:
            return db.execute('DELETE FROM templates WHERE id=?', (template_id,)).rowcount > 0


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()
