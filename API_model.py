from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle
import numpy as np

#initialize flask
app = Flask(__name__)

#load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def home():
    return "Loaded machine learning model"

@app.route('/predict',methods=['POST')
def predict(): 
    # Get JSON Request
    feat_data = request.json
    # Convert JSON request to Pandas DataFrame
    df = pd.DataFrame(feat_data)
    # Match Column Na,es
    df = df.reindex(columns=col_names)
    # Get prediction
    prediction = list(model.predict(df))
    # Return JSON version of Prediction
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    # load model, feat columns
    model = joblib.load("final_model.pkl") 
    col_names = joblib.load("column_names.pkl") 

    app.run(debug=True)

     
