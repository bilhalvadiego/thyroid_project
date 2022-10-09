#!/usr/bin/env python
# coding: utf-8

# In[31]:


# import pickle

# model = pickle.load(open('xgb_clf.pkl','rb'))



# In[32]:


# import pandas as pd
# df = pd.DataFrame([[ 64.  ,   1.  ,   1.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
#           0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
#           2.2 ,   0.  , 111.  ,   0.97, 114.  ,   1.  ]])


# In[35]:


# import pandas as pd
# df = pd.DataFrame([[54.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
#          1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  7.2 ,  0.  ,
#         74.  ,  0.88, 84.]])


# In[36]:


# model.predict(df)


# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('xgb_clf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([features], columns=model.get_booster().feature_names)
    
    # print(features)
    
    # return features
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    output = 'maligno' if output == 1 else 'benigno'
    
    return render_template('index.html', prediction_text='O paciente tem cancer {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    
    age = data['age']
    sex = data['sex']
    on_thyroxine = data['on_thyroxine']
    query_on_thyroxine = data['query_on_thyroxine']
    on_antithyroid_meds = data['on_antithyroid_meds']
    sick = data['sick'],
    pregnant = data['pregnant']
    thyroid_surgery = data['thyroid_surgery']
    I131_treatment = data['I131_treatment']
    query_hypothyroid = data['query_hypothyroid']
    query_hyperthyroid = data['query_hyperthyroid']
    lithium = data['lithium']
    goitre = data['goitre']
    tumor = data['tumor']
    hypopituitary = data['hypopituitary']
    psych = data['psych']
    TSH = data['TSH']
    T3 = data['T3']
    TT4 = data['TT4']
    T4U = data['T4U']
    FT = data['FTI']
    
    data_dict = {
        'age':age,
        'sex':sex,
        'on_thyroxine':on_thyroxine,
        'query_on_thyroxine':query_on_thyroxine,
        'on_antithyroid_meds':on_antithyroid_meds,
        'sick':sick,
        'pregnant':pregnant,
        'thyroid_surgery':thyroid_surgery,
        'I131_treatment':I131_treatment,
        'query_hypothyroid':query_hypothyroid,
        'query_hyperthyroid':query_hyperthyroid,
        'lithium':lithium,
        'goitre':goitre,
        'tumor':tumor,
        'hypopituitary':hypopituitary,
        'psych':psych,
        'TSH':TSH,
        'T3':T3,
        'TT4':TT4,
        'T4U':T4U,
        'FTI':FTI
    }
    
    data = {k:[v] for k,v in data_dict.items()}
    prediction = model.predict(pd.DataFrame(data))
    
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

