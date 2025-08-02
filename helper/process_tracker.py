import sqlite3
import psutil
from datetime import datetime

class ProcessStatsTracker:
    def __init__(self,
                 current_db='current_day.db',
                 overall_db='overall.db',
                 history_db='history.db'):
        # Setup database connections
        self.current_conn = sqlite3.connect(current_db)
        self.overall_conn = sqlite3.connect(overall_db)
        self.history_conn = sqlite3.connect(history_db)

        self.current_cursor = self.current_conn.cursor()
        self.overall_cursor = self.overall_conn.cursor()
        self.history_cursor = self.history_conn.cursor()

        # In-memory process stats
        self.process_stats = {}
        print("creating databases")
        # Create tables if not exist
        self.setup_databases()
        print("database creation completed")

    def setup_databases(self):
        self.current_cursor.execute("""
        CREATE TABLE IF NOT EXISTS process_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            process_name TEXT,
            memory_usage_mb REAL,
            time_in_top5 INTEGER
        )
        """)

        self.overall_cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_process_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            process_name TEXT,
            total_time_in_top5_sec INTEGER,
            average_memory_mb REAL
        )
        """)

        self.history_cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            top_process_name TEXT,
            total_time_in_top5_sec INTEGER,
            average_memory_mb REAL
        )
        """)

        self.current_conn.commit()
        self.overall_conn.commit()
        self.history_conn.commit()

    def log_top_processes(self):
        # Get all processes with pid, name, and memory usage
        processes = [(p.info['pid'], p.info['name'], p.info['memory_info'].rss / 1024 / 1024)
                     for p in psutil.process_iter(['pid', 'name', 'memory_info'])]

        # Sort and pick top 5 by memory usage
        top5 = sorted(processes, key=lambda x: x[2], reverse=True)[:5]

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for pid, name, mem_usage in top5:
            if pid not in self.process_stats:
                self.process_stats[pid] = {'name': name, 'time_in_top5': 0, 'total_memory': 0, 'samples': 0}

            self.process_stats[pid]['time_in_top5'] += 1
            self.process_stats[pid]['total_memory'] += mem_usage
            self.process_stats[pid]['samples'] += 1

            avg_mem = self.process_stats[pid]['total_memory'] / self.process_stats[pid]['samples']

            self.current_cursor.execute("""
            INSERT INTO process_stats (timestamp, process_name, memory_usage_mb, time_in_top5)
            VALUES (?, ?, ?, ?)
            """, (timestamp, name, mem_usage, self.process_stats[pid]['time_in_top5']))

        self.current_conn.commit()

    def flush_daily_stats_to_overall(self):
        date_today = datetime.now().strftime('%Y-%m-%d')

        for pid, stats in self.process_stats.items():
            avg_mem = stats['total_memory'] / stats['samples']

            self.overall_cursor.execute("""
            INSERT INTO daily_process_stats (date, process_name, total_time_in_top5_sec, average_memory_mb)
            VALUES (?, ?, ?, ?)
            """, (date_today, stats['name'], stats['time_in_top5'], avg_mem))

        self.overall_conn.commit()

    def save_daily_summary_to_history(self):
        date_today = datetime.now().strftime('%Y-%m-%d')

        if not self.process_stats:
            print("No data to summarize.")
            return

        # Find top process by time_in_top5
        top_pid = max(self.process_stats.items(), key=lambda x: x[1]['time_in_top5'])[0]
        top_process = self.process_stats[top_pid]
        avg_mem = top_process['total_memory'] / top_process['samples']

        self.history_cursor.execute("""
        INSERT INTO daily_summary (date, top_process_name, total_time_in_top5_sec, average_memory_mb)
        VALUES (?, ?, ?, ?)
        """, (date_today, top_process['name'], top_process['time_in_top5'], avg_mem))

        self.history_conn.commit()

    def cleanup_history_limit(self):
        # Keep only last 10 days of history
        self.history_cursor.execute("""
        DELETE FROM daily_summary
        WHERE date NOT IN (
            SELECT date FROM daily_summary
            ORDER BY date DESC LIMIT 10
        )
        """)
        self.history_conn.commit()

    # Fetch methods
    def fetch_current_day_stats(self):
        self.current_cursor.execute("SELECT process_name, SUM(time_in_top5) FROM process_stats GROUP BY process_name")
        data = self.current_cursor.fetchall()
        return data

    def fetch_overall_stats(self):
        self.overall_cursor.execute("SELECT process_name, SUM(total_time_in_top5_sec) FROM daily_process_stats GROUP BY process_name")
        data = self.overall_cursor.fetchall()
        return data

    def fetch_daily_history_stats(self):
        self.history_cursor.execute("SELECT date, total_time_in_top5_sec FROM daily_summary ORDER BY date")
        data = self.history_cursor.fetchall()
        return data

    # Optional: Close connections on cleanup
    def close(self):
        self.current_conn.close()
        self.overall_conn.close()
        self.history_conn.close()
