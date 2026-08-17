import duckdb, sys

db = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars(r"%LOCALAPPDATA%\hermes\hybrid_memory.duckdb")
con = duckdb.connect(db, read_only=True)
tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()]
print(f"DB: {db}")
print("TABLES:", tables)
for t in tables:
    try:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t}: {n} rows")
    except Exception as e:
        print(f"  {t}: ERROR {e}")
con.close()
