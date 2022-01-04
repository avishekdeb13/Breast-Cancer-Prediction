import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 2:
        a = "NO"
    else:
        a = "YES"

    return render_template('index.html', prediction_text=f"Breast Cancer Detected? :  {a}")

if __name__ == "__main__":
    app.run(debug=True)