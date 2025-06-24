from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from detect_pick import analyze_pick_action
from db_utils import init_db, write_one_result_to_db
import sqlite3

app = Flask(__name__)
init_db()


@app.route('/keypoints', methods=['POST'])
def receive_keypoints():
    data = request.get_json()
    keypoints_buffer = data.get("keypoints_buffer", {})
    h = data.get("shape_h", None)
    w = data.get("shape_w", None)
    pick_results = []

    pick_results = analyze_pick_action(keypoints_buffer, h, w)

    return jsonify({
        "picked_ids": pick_results
    })

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.get_json()
    track_id = data.get("track_id")
    gender = data.get("gender")
    age = data.get("age")
    race = data.get("race")
    behavior = data.get("behavior")
    obj_list = data.get("objects", [])

    write_one_result_to_db(track_id, gender, age, race, behavior, obj_list)

    return jsonify({"status": "ok"})


@app.route('/records', methods=['GET'])
def get_all_records():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT track_id, gender, age, race, behavior, objects, timestamp FROM behavior_records ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    # 转为字典列表
    results = []
    for row in rows:
        results.append({
            "track_id": row[0],
            "gender": row[1],
            "age": row[2],
            "race": row[3],
            "behavior": row[4],
            "objects": row[5],
            "timestamp": row[6]
        })

    return jsonify(results)

@app.route('/records_page')
def records_page():
    return render_template("records.html")

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    save_name = request.form.get('save_name', file.filename)

    if not file:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    filename = secure_filename(save_name)
    save_dir = 'uploaded_images'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    file.save(save_path)
    return jsonify({"status": "ok", "saved_as": filename})
