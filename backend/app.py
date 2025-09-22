from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend to talk to backend

# Simple test endpoint
@app.route('/')
def hello():
    return jsonify({"message": "Restaurant Popularity Predictor API is running!"})

# Main prediction endpoint (we'll build this out gradually)
# ROUTE 3: Main prediction endpoint - NOW WITH REAL ML!
@app.route('/predict', methods=['POST'])
def predict_popularity():
    data = request.get_json()
    
    # For now, just return dummy data
    restaurant_name = data.get('restaurant_name', 'Unknown')
    location = data.get('location', 'Unknown')
    
    # Dummy prediction (we'll replace this with real ML later)
    fake_prediction = {
        "restaurant": restaurant_name,
        "location": location,
        "popularity_score": 7.5,
        "growth_prediction": "High",
        "confidence": 0.85
    }
    
    return jsonify(fake_prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)