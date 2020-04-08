# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:27:43 2020

@author: akaniyamparambil
"""

import requests
import pickle
import numpy as np
import json
import pandas as pd

file = open('./Pickle/Dictionary_Sample.p','rb')
Dictionary_legend=pickle.load( file )
file.close()
Cat_vars=['Shipment Category', 'Description', 'Date', 'Partial']
def Create_Samples():
    Sample_dict={}
    for f,f1 in Dictionary_legend.items():
        if f in Cat_vars:
            #print(2)
            #print(f,len(f1))
            Sample_dict[f]=f1[np.random.randint(0,len(f1))]
        else:
            #print(f1[0],f1[1])
            Sample_dict[f]=np.random.randint(f1[1],f1[0])
    Sample_dict.pop("Target") 
    print("Success 1234567890")
       
    Popped=Sample_dict.pop("Partial")
    Sample_dict["Partial"]=Popped
    Sample_dict["Date"]=str(Sample_dict["Date"])
    print(pd.Timestamp(Sample_dict["Date"]))
    return Sample_dict
  

# =============================================================================
# 
# sample=json.dumps(Create_Samples())
# 
# url = "http://localhost:3000/Predict"
# r= requests.post(url,json=sample)
# print(r.text)
# =============================================================================
