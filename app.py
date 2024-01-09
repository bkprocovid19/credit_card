from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
from pytz import timezone
import joblib
import os
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

# Model for Prediction History
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    features = db.Column(db.String(255))
    result = db.Column(db.String(50))


csv_file_path = 'data\S-FFSDU.xlsx'

# Đọc file CSV và lưu vào một DataFrame
data = pd.read_excel(csv_file_path)
# Lọc dữ liệu chỉ chọn những dòng có nhãn là 0 hoặc 1
filtered_data = data[data['Labels'].isin([0, 1])]

# Chia dữ liệu thành features (X) và labels (y)
y = filtered_data['Labels'].astype(np.int32)
X = filtered_data.drop('Labels', axis=1)

def model_file_exists(filename):
    return os.path.isfile(filename)


# Function to train and save the model
def train_and_save_model():

# Tạo và huấn luyện mô hình CatBoost
    model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='Logloss')
    model.fit(X, y, cat_features=['Source', 'Target', 'Location', 'Type'])
    # Lưu mô hình vào một file
    model_filename = 'cat_model.joblib'
    joblib.dump(model, model_filename)
    return model, model_filename

model_filename = 'cat_model.joblib'



if model_file_exists(model_filename):
    # Load the model if it exists
    trained_model = joblib.load(model_filename)
else:
    # Train the model if it doesn't exist
    trained_model, model_filename = train_and_save_model()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_fraud_knn', methods=['POST'])
def detect_fraud_knn():
    # Extracting values from the form
    time = np.float32(float(request.form.get('time')))
    source_account = request.form.get('source_account')
    destination_account = request.form.get('destination_account')
    transaction_amount = np.float32(float(request.form.get('transaction_amount')))
    location = request.form.get('location')
    transaction_type = request.form.get('transaction_type')

    feature = np.array([[time,source_account,destination_account,transaction_amount,location,transaction_type]])
    feature1 = pd.DataFrame(feature, columns=X.columns)

    # Predict using the trained KNN model
    prediction = trained_model.predict(feature1)
    probabilities = trained_model.predict_proba(feature1)
    fraud_probability_str = f'Xác suất gian lận: {round(probabilities[0][1],3)}'
    # Save prediction history
    save_prediction_history(feature, fraud_probability_str)

    result = {'result': 'Gian lận' if prediction[0] == 1 else 'Hợp pháp', 'fraud_probability': fraud_probability_str}
    return jsonify(result)

def save_prediction_history(feature, result):
    with app.app_context():  # Bao quanh hoạt động liên quan đến cơ sở dữ liệu
        features_str = ", ".join(map(str, feature.flatten()))
        new_prediction = PredictionHistory(features=features_str, result = result)
        db.session.add(new_prediction)
        db.session.commit()

@app.route('/prediction_history')
def prediction_history():
    with app.app_context():  # Bao quanh hoạt động liên quan đến cơ sở dữ liệu
        predictions = PredictionHistory.query.all()
    return render_template('history.html', predictions=predictions)



@app.template_filter('to_local_time')
def to_local_time(value):
    if value is None:
        return ""
    if isinstance(value, str):
        value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    vn_tz = timezone('Asia/Ho_Chi_Minh')  # Múi giờ Việt Nam
    value = value.replace(tzinfo=timezone('UTC')).astimezone(vn_tz)
    return value.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    with app.app_context():  # Bao quanh tạo bảng cơ sở dữ liệu
        db.create_all()
    app.run(debug=True)