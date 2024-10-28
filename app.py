from flask import Flask, request, render_template
import numpy as np
import pickle
import requests

# Load model and scaler
Model = pickle.load(open('gnb_model.pkl', 'rb'))
MinMaxScaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)


def fetch_random_image(crop_name):
    # Example using Unsplash API
    api_url = f"https://api.unsplash.com/search/photos?query={crop_name}&client_id=4FtxD6-mTiqC0SWpX7AZEVLGZ21YBwnoU8QnVW4_XZ8"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            # Return the URL of a random image
            return data['results'][0]['urls']['regular']
    return None  # Return None if no images are found

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    return render_template("predict.html")


@app.route("/submit", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_prediction = np.array(feature_list).reshape(1, -1)

    scaled_features = MinMaxScaler.transform(single_prediction)
    prediction = Model.predict(scaled_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
                 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
                 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
                 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
                 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there."
        # Fetch a random image based on the crop
        image_url = fetch_random_image(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        image_url = None  # No image if no prediction is made

    return render_template('predict.html', result=result, image_url=image_url)

# Python main
if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0", port=5000)
