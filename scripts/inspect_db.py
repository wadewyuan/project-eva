#!/usr/bin/env python3
"""Quick SQLite inspector for Eva's database.

Usage:
    uv run python scripts/inspect_db.py              # show all tables overview
    uv run python scripts/inspect_db.py --profile    # show user_profile
    uv run python scripts/inspect_db.py --memories   # show memories
    uv run python scripts/inspect_db.py "SELECT * FROM user_profile"
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "eva.db"


def run_sql(conn: sqlite3.Connection, sql: str) -> list:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description] if cur.description else []
    return columns, rows


MAX_CELL_WIDTH = 80


def _fmt(val) -> str:
    s = str(val)
    if len(s) > MAX_CELL_WIDTH:
        return s[:MAX_CELL_WIDTH - 1] + "…"
    return s


def print_table(columns: list[str], rows: list[tuple]) -> None:
    if not columns:
        print(f"  ({len(rows)} rows, no columns)")
        return

    # Format all values first
    formatted = [[_fmt(v) for v in row] for row in rows]

    # Calculate column widths
    widths = [len(c) for c in columns]
    for row in formatted:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Cap total width
    widths = [min(w, MAX_CELL_WIDTH) for w in widths]

    # Print header
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    print("  " + header)
    print("  " + "-+-".join("-" * w for w in widths))

    # Print rows
    for row in formatted:
        line = " | ".join(v.ljust(w) for v, w in zip(row, widths))
        print("  " + line)

    print(f"\n  ({len(rows)} rows)")


def show_overview(conn: sqlite3.Connection) -> None:
    # List tables
    _, tables = run_sql(conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [t[0] for t in tables if not t[0].startswith("sqlite_") and not t[0].startswith("memories_fts")]

    for name in table_names:
        print(f"\n=== {name} ===")
        columns, rows = run_sql(conn, f"SELECT * FROM {name} LIMIT 20")
        print_table(columns, rows)


def show_profile(conn: sqlite3.Connection) -> None:
    print("\n=== user_profile ===")
    columns, rows = run_sql(
        conn, "SELECT fact_type, fact_key, fact_value, confidence, updated_at FROM user_profile ORDER BY fact_type"
    )
    print_table(columns, rows)


def show_memories(conn: sqlite3.Connection) -> None:
    print("\n=== memories ===")
    columns, rows = run_sql(
        conn,
        "SELECT id, fact_text, category, importance_score, created_at, source_context "
        "FROM memories ORDER BY created_at DESC LIMIT 20",
    )
    print_table(columns, rows)


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        if len(sys.argv) < 2:
            show_overview(conn)
        elif sys.argv[1] == "--profile":
            show_profile(conn)
        elif sys.argv[1] == "--memories":
            show_memories(conn)
        else:
            sql = sys.argv[1]
            columns, rows = run_sql(conn, sql)
            print_table(columns, rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
