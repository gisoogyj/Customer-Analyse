# db_utils.py

import sqlite3
from datetime import datetime

DB_PATH = 'analytics.db'  # 默认数据库路径，可按需改为绝对路径

# 初始化数据库，创建表结构（只需执行一次）
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 行为记录表
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

    # 人群统计表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crowd_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_range TEXT,
            total_count INTEGER,
            male_count INTEGER,
            female_count INTEGER,
            age_0_2 INTEGER,
            age_3_9 INTEGER,
            age_10_19 INTEGER,
            age_20_29 INTEGER,
            age_30_39 INTEGER,
            age_40_49 INTEGER,
            age_50_59 INTEGER,
            age_60_69 INTEGER,
            age_70_plus INTEGER,
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

def write_crowd_stats_to_db(time_range, total, male, female, age_stats, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 按 age_map 顺序插入
    age_map = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    age_values = [age_stats.get(k, 0) for k in age_map]

    cursor.execute('''
        INSERT INTO crowd_statistics (
            time_range, total_count, male_count, female_count,
            age_0_2, age_3_9, age_10_19, age_20_29, age_30_39,
            age_40_49, age_50_59, age_60_69, age_70_plus, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (time_range, total, male, female, *age_values, timestamp))

    conn.commit()
    conn.close()
