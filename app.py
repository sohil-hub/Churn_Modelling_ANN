from os import name
from flask import Flask, render_template, request
import sklearn
import joblib
import requests
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Country         = request.form['Country']
        CreditScore     = int(request.form['CreditScore'])
        Gender          = int(request.form['Gender'])
        Age             = int(request.form['Age'])
        Tenure          = int(request.form['Tenure'])
        Balance         = float(request.form['Balance'])
        NumberOfProducts = int(request.form['NumberOfProducts'])
        HasCrCard       = int(request.form['hasCrCard'])
        IsActiveMember  = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['Salary'])

        if Country=='France':
            c1, c2, c3 = 1, 0, 0
        elif Country=='Spain':
            c1, c2, c3 = 0, 0, 1
        else:
            c1, c2, c3 = 0, 1, 0
        
        input = np.array([c1,c2, c3, CreditScore, Gender, Age, Tenure, Balance, NumberOfProducts, HasCrCard, IsActiveMember, EstimatedSalary])
        print(input)

        scaler = joblib.load('scaler')
        model = tf.keras.models.load_model('classifier')
        prediction = str((model.predict(scaler.transform([input])) > 0.5)[0][0])
        
        print(model.predict(scaler.transform([input])))
        print(prediction)
        if prediction == 'False':
            return render_template('index.html',prediction_text="The customer will leave the bank")
        else:
            return render_template('index.html',prediction_text="The customer will not leave the bank")
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)


#France 100, spain 001, germany 010
#male 1 female 0