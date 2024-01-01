from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
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
    result = db.Column(db.String(20))


csv_file_path = 'data\S-FFSD.csv'

# Đọc file CSV và lưu vào một DataFrame
data = pd.read_csv(csv_file_path)
filtered_data = data[data['Labels'].isin([0, 1])]

# Convert data to NumPy array
data_np = np.array(filtered_data)

# Huấn luyện encoder cho biến hạng mục 'source_account'
source_account_encoder = OneHotEncoder(handle_unknown='ignore')
source_account_encoded = source_account_encoder.fit_transform(data_np[:, [1,2,4,5]]).toarray()

# Lấy features dạng số (numerical) từ các cột không phải biến hạng mục
numerical_features = data_np[:, [0, 3]].astype(np.float32)

# Nối features dạng số với features đã được mã hóa
final_features = np.concatenate([numerical_features, source_account_encoded], axis=1)

# Target variable
y = data_np[:, -1].astype(np.int32)


def model_file_exists(filename):
    return os.path.isfile(filename)


# Function to train and save the model
def train_and_save_model():
    # Huấn luyện mô hình KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(final_features, y)

    # Lưu mô hình vào một file
    model_filename = 'knn_model.joblib'
    joblib.dump(knn_model, model_filename)

    return knn_model, model_filename

model_filename = 'knn_model.joblib'



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

    # Huấn luyện encoder cho biến hạng mục 'source_account'
    source_account_encoded1 = source_account_encoder.transform([[source_account,destination_account,location,transaction_type]]).toarray()

    # Preprocess the input features
    features = np.array([[time, transaction_amount]])

    # Nối features dạng số với features đã được mã hóa
    final_features1 = np.concatenate([features, source_account_encoded1], axis=1)

    # Predict using the trained KNN model
    prediction = trained_model.predict(final_features1)

    # Save prediction history
    save_prediction_history(feature, prediction)

    return jsonify({'result': 'Gian lận' if prediction == 1 else 'Hợp pháp'})

def save_prediction_history(feature, result):
    with app.app_context():  # Bao quanh hoạt động liên quan đến cơ sở dữ liệu
        features_str = ", ".join(map(str, feature.flatten()))
        new_prediction = PredictionHistory(features=features_str, result='Gian lận' if result == 1 else 'Hợp pháp')
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