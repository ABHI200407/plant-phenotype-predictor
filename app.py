import os
import json
from datetime import timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tasks import predict_lite_task, process_csv_task, celery 

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
# EXTEND SESSION TIME: Token lasts for 24 hours
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24) 

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

db = SQLAlchemy(app)
jwt = JWTManager(app)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    def set_password(self, password): 
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password): 
        return check_password_hash(self.password_hash, password)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    task_id = db.Column(db.String(100), nullable=False)
    features = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="Pending")
    result = db.Column(db.Float, nullable=True)

# Create Tables
with app.app_context():
    db.create_all()

# --- ROUTES ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data.get('username') or not data.get('password') or not data.get('email'):
        return jsonify({"error": "Missing fields"}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "User already exists"}), 400

    new_user = User(username=data['username'], email=data['email'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    if user and user.check_password(data.get('password')):
        token = create_access_token(identity=str(user.id))
        return jsonify({"access_token": token, "username": user.username})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/user/update', methods=['PUT'])
@jwt_required()
def update_user():
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    data = request.get_json()

    # Update Username
    if 'username' in data and data['username']:
        if data['username'] != user.username and User.query.filter_by(username=data['username']).first():
             return jsonify({"error": "Username already taken"}), 400
        user.username = data['username']
    
    # Update Email
    if 'email' in data and data['email']:
        user.email = data['email']
    
    # Update Password
    if 'password' in data and data['password']:
        user.set_password(data['password'])
    
    db.session.commit()
    return jsonify({"message": "Profile updated successfully", "username": user.username})

@app.route('/user/history', methods=['GET'])
@jwt_required()
def get_user_history():
    current_user_id = int(get_jwt_identity())
    
    # Get last 5 records for this specific user
    history = PredictionHistory.query.filter_by(user_id=current_user_id).order_by(PredictionHistory.id.desc()).limit(5).all()
    
    history_list = []
    for r in history:
        history_list.append({
            "id": r.id,
            "task_id": r.task_id,
            "type": "Batch" if "Batch" in r.features else "Lite",
            "status": r.status,
            "result": r.result
        })
    return jsonify(history_list)

@app.route('/predict/lite', methods=['POST'])
@jwt_required()
def predict_lite():
    current_user_id = int(get_jwt_identity())
    data = request.get_json()
    features = data.get('features', [])
    
    task = predict_lite_task.delay(features)
    
    new_record = PredictionHistory(
        user_id=current_user_id,
        task_id=task.id,
        features=json.dumps(features),
        status="Processing"
    )
    db.session.add(new_record)
    db.session.commit()
    
    return jsonify({"status": "Lite Task submitted", "task_id": task.id})

@app.route('/predict/batch', methods=['POST'])
@jwt_required()
def predict_batch():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    
    task = process_csv_task.delay(filepath, user.email)
    
    new_record = PredictionHistory(
        user_id=current_user_id,
        task_id=task.id,
        features="Batch CSV Upload",
        status="Processing"
    )
    db.session.add(new_record)
    db.session.commit()
    
    return jsonify({"status": "Batch Processing Started", "task_id": task.id})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    # Check Celery Task State
    task = celery.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {"state": "Pending", "status": "Processing..."}
    elif task.state == 'SUCCESS':
        response = {"state": "Success", "result": task.result}
    elif task.state == 'FAILURE':
        response = {"state": "Failure", "status": str(task.info)}
    else:
        response = {"state": task.state}
        
    return jsonify(response)

@app.route('/admin/stats', methods=['GET'])
@jwt_required()
def get_admin_stats():
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)

    # Security Check
    if user.username != 'admin':
        return jsonify({"error": "Unauthorized: Admins only"}), 403

    total_users = User.query.count()
    total_predictions = PredictionHistory.query.count()
    
    # Calculate counts for Charts
    lite_count = PredictionHistory.query.filter(PredictionHistory.features.notlike('%Batch%')).count()
    batch_count = PredictionHistory.query.filter(PredictionHistory.features.like('%Batch%')).count()
    
    recent = PredictionHistory.query.order_by(PredictionHistory.id.desc()).limit(5).all()
    recent_list = []
    for r in recent:
        u = User.query.get(r.user_id)
        recent_list.append({
            "id": r.id,
            "user": u.username if u else "Unknown",
            "type": "Batch" if "Batch" in r.features else "Lite",
            "status": r.status
        })

    return jsonify({
        "total_users": total_users,
        "total_predictions": total_predictions,
        "chart_data": [
            {"name": "Lite Model", "value": lite_count},
            {"name": "Batch Model", "value": batch_count}
        ],
        "recent_activity": recent_list
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)