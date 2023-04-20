import numpy as np
from flask import Flask,request,jsonify,render_template
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
app = Flask(__name__)
Diabetes=pickle.load(open('models/model.pkl','rb'))
scaler=pickle.load(open('models/Diabetes_scaler.pkl','rb'))

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')
@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        print(data)
        result=Diabetes.predict(data)
      

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')



    
if __name__=="__main__":
    app.run(host="0.0.0.0" )