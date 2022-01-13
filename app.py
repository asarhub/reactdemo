import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('instagram.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==1):
        return render_template('instagram.html', prediction_text='yes, you have probably heart disease and better u consult a doctor'.format(output))
    else:
        return render_template('instagram.html', prediction_text='No ,You dont have heart disease'.format(output))

if __name__ == "__main__":
    app.run(debug=True)