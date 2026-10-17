import hashlib
import json
import sqlite3
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class ReportStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, source TEXT, identity TEXT, payload TEXT, PRIMARY KEY(kind, source, identity))')
        self.db.execute('CREATE TABLE IF NOT EXISTS cursors (path TEXT PRIMARY KEY, position INTEGER, head BLOB, tail BLOB)')
        self.db.commit()

    def put(self, kind, source, key, payload):
        encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False)
        old = self.db.execute('SELECT payload FROM records WHERE kind=? AND source=? AND identity=?', (kind, source, key)).fetchone()
        if old is not None and old[0] == encoded:
            return False
        self.db.execute('INSERT INTO records VALUES (?, ?, ?, ?) ON CONFLICT(kind, source, identity) DO UPDATE SET payload=excluded.payload',
            (kind, source, key, encoded))
        return True

    def records(self, kind):
        return [(source, key, json.loads(value)) for source, key, value in
            self.db.execute('SELECT source, identity, payload FROM records WHERE kind=? ORDER BY rowid', (kind,))]

    def get(self, kind, source, key):
        row = self.db.execute('SELECT payload FROM records WHERE kind=? AND source=? AND identity=?', (kind, source, key)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self, kind):
        return self.db.execute('SELECT count(*) FROM records WHERE kind=?', (kind,)).fetchone()[0]

    def cursor(self, path):
        return self.db.execute('SELECT position, head, tail FROM cursors WHERE path=?', (str(path),)).fetchone()

    def move(self, path, position, head, tail):
        self.db.execute('INSERT INTO cursors VALUES (?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET position=excluded.position, head=excluded.head, tail=excluded.tail',
            (str(path), position, head, tail))
