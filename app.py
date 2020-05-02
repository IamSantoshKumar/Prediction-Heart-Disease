from sklearn.externals import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")
database={'nachi':'123','james':'aac','karthik':'asdsf'}

@app.route('/predict',methods=['POST','GET'])
def predict():
    Age=request.form['Age']
    Sex=request.form['Sex']
    Cp=request.form['Cp']
    Trestbps=request.form['Trestbps']
    Chol=request.form['Chol']
    Fbs=request.form['Fbs']
    Restecg=request.form['Restecg']
    Thalach=request.form['Thalach']
    Exang=request.form['Exang']
    Oldpeak=request.form['Oldpeak']
    Slope=request.form['Slope']
    Ca=request.form['Ca']
    Thal=request.form['Thal']
    df=pd.DataFrame({'Age':Age,'Sex':Sex,'Cp':Cp,'Trestbps':Trestbps,'Chol':Chol,'Fbs':Fbs,'Restecg':Restecg,'Thalach':Thalach,'Exang':Exang,'Oldpeak':Oldpeak,'Slope':Slope,'Ca':Ca,'Thal':Thal},index=[0])
    for cols in df.columns:
        if cols not in ['Oldpeak']:
            df[cols]=df[cols].astype('int64')
        else:
            df[cols]=df[cols].astype('float32')
    #df['Embarked']=df['Embarked'].astype('int64')
    knn_from_joblib = joblib.load('HeartDisease_knn.sav') 
    pred=knn_from_joblib.predict(df)
    pred_proba=knn_from_joblib.predict_proba(df)
    pred_proba_f=round(np.max(pred_proba[0])*100,2)
    if pred[0]==1:
        fin=str(pred_proba_f)+'%'+' of chances having Heart Disease'
    else:
        fin=str(pred_proba_f)+'%'+' Chances not having Heart Disease'
    #print(pred[0],fin,pred_proba_f,pred_proba_f)
    return render_template('index.html',name=fin,pred=pred[0])

if __name__ == '__main__':
    app.run()