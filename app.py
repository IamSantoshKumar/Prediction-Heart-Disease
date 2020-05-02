from flask import Flask,request,render_template
from flask import Flask,request,render_template
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
%matplotlib inline

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