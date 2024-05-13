from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from google.cloud import storage
import csv

app = Flask(__name__)

# Configure Flask application to use SQLAlchemy and connect to the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sample.db'
db = SQLAlchemy(app)

# Define SQLAlchemy models for structured data
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

# Authenticate with Google Cloud Storage (replace with actual credentials)
storage_client = storage.Client.from_service_account_json('path/to/credentials.json')

# Define route to handle requests for structured data
@app.route('/structured_data', methods=['GET'])
def get_structured_data():
    # Query the database using SQLAlchemy
    users = User.query.all()
    # Convert query results to JSON
    structured_data = [{'id': user.id, 'username': user.username, 'email': user.email} for user in users]
    return jsonify(structured_data)

# Define route to handle requests for unstructured data
@app.route('/unstructured_data', methods=['GET'])
def get_unstructured_data():
    # Access cloud storage service (replace with actual logic to access unstructured data)
    bucket = storage_client.get_bucket('your-bucket-name')
    blobs = bucket.list_blobs()
    # Convert cloud storage data to JSON
    unstructured_data = [blob.name for blob in blobs]
    return jsonify(unstructured_data)

# Define route to handle ingestion of data from CSV file
@app.route('/ingest_csv', methods=['POST'])
def ingest_csv():
    try:
        # Read CSV file from request
        csv_file = request.files['csv_file']
        # Assuming CSV file has headers and comma-separated values
        reader = csv.reader(csv_file)
        for row in reader:
            user = User(username=row[0], email=row[1])
            db.session.add(user)
        db.session.commit()
        return 'CSV data ingested successfully!'
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
