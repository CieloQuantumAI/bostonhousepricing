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
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))#need to convert the values into list
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))#new data that is transformed
    output=regmodel.predict(new_data)#predicting the values
    print(output[0])#gettingt he first value of a two dimensional array
    return jsonify(output[0])#returning the output in json format

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]#getting the values from the form, converting them into float and storing them in a list
    final_input=scalar.transform(np.array(data).reshape(1, -1))#reshaping the data and transforming it
    print(final_input)
    output=regmodel.predict(final_input)[0]#predicting the values, first value 0 will give me te result (this is a two dimensional array)
    return render_template("home.html", prediction_text="The House price prediction is {}".format(output))# reder template is important in flask, this will render an HTML page




if __name__ == "__main__":
    app.run(debug=True)



