# db_utils.py

import sqlite3
from datetime import datetime

DB_PATH = 'results.db'  # 默认数据库路径，可按需改为绝对路径

# 初始化数据库，创建表结构（只需执行一次）
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS behavior_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT,
            gender TEXT,
            age TEXT,
            race TEXT,
            behavior TEXT,
            objects TEXT,
            timestamp TEXT
        )
    ''')

    conn.commit()
    conn.close()

# 写入一条分析记录（来自树莓派的行为识别结果）
def write_one_result_to_db(track_id, gender, age, race, behavior, obj_list, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    obj_str = ','.join(obj_list) if obj_list else ""

    cursor.execute('''
        INSERT INTO behavior_records (
            track_id, gender, age, race, behavior, objects, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (track_id, gender, age, race, behavior, obj_str, timestamp))

    conn.commit()
    conn.close()
