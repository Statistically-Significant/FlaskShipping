# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:50:09 2020

@author: akaniyamparambil
"""

from MakeRequests import Create_Samples
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json
import pandas as pd


app = Flask(__name__,template_folder='templates')

file = open('./Pickle/Poc_Model_Regression_90.p', 'rb')
Model = pickle.load(file)
columnTransformer = pickle.load(file)
file.close()



@app.route("/Home")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route("/Predict", methods=['POST','GET'])
def predict():
    if request.method=='POST':
            print("You have reached YOUR DESTINATION")
       
            X_vars=request.form.to_dict()
            print(X_vars)
            print(type(X_vars))
            #X_vars=request.get_json()
            #json1_data = json.dumps(X_vars)
            #print(type(json1_data))
            #X_vars.pop("Target")
            req=[]
            counter=1
            for c1,c2 in X_vars.items():
                if counter ==3:
                    req.append(pd.Timestamp('2019-03-11'))
                    
                req.append(c2)
                counter+=1
            print(req,req[2],type(req[2]))
            #req[2]=pd.Timestamp(req[2])
            required=np.array(req).reshape(1,-1)
            print(columnTransformer)
            required1=columnTransformer.transform(required).toarray()
            #print(required1)
            #break
            pred=Model.predict(required1)
            #dict={}
            #dict["predictions"]=pred
            print(f"The predictions are {pred[0]}")
            #return render_template('Main.html',sentiment=jsonify(pred[0]))
            return jsonify(Pred=pred[0],X1=X_vars)

  
    return render_template('Main.html',sentiment='')


    
@app.route('/background_process_test')
def background_process_test():
    print("You have reached")
    Make_vals=Create_Samples()
    print(Make_vals)
    return (Make_vals)


if __name__ == '__main__':
    app.run(debug=True, port=3000)