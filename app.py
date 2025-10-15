import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model
regmodel=pickle.load(open('regmodel.pkl', 'rb'))#loading the model
scalar= pickle.load(open('scaling.pkl', 'rb'))#loading the scaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data=request.json['data']#we are passign as JSON and are getting the dictionary values
    print(np.array(list(data.values())).reshape(1, -1))#need to convert the values into list
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))#new data that is transformed
    output=regmodel.predict(new_data)#predicting the values
    print(output[0])#gettingt he first value of a two dimensional array
    return jsonify(output[0])#returning the output in json format

if __name__ == "__main__":
    app.run(debug=True)



