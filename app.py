import pickle
from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained models
kmeans = joblib.load("kmeans_model.pkl")
gb_model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to make predictions
def predict_clusters_and_target(features_scaled, kmeans, gb_model):
    # Predict KMeans clusters
    clusters = kmeans.predict(features_scaled)

    # Predict target variable using Gradient Boosting model
    target_prediction = gb_model.predict(features_scaled)
    target_prediction_proba = gb_model.predict_proba(features_scaled)

    return clusters, target_prediction, target_prediction_proba

@app.route('/')
def index():
    # Default values

    ## for class 0
    # default_values = {
    #     'education': 'Graduation',
    #     'income': 62513,
    #     'kidhome': 0,
    #     'teenhome': 1,
    #     'recency': 16,
    #     'wines': 520,
    #     'fruits': 42,
    #     'meat': 98,
    #     'fish': 0,
    #     'sweets': 42,
    #     'gold': 14,
    #     'num_deals_purchases': 2,
    #     'num_web_purchases': 6,
    #     'num_catalog_purchases': 4,
    #     'num_store_purchases': 10,
    #     'num_web_visits_month': 6,
    #     'accepted_cmp3': 0,
    #     'accepted_cmp4': 0,
    #     'accepted_cmp5': 0,
    #     'accepted_cmp1': 0,
    #     'accepted_cmp2': 0,
    #     'complain': 0,
    #     'response': 0,
    #     'time_enrolled_days': 479,
    #     'age': 57,
    #     'spent': 716,
    #     'living_with': 'Partner',
    #     'children': 1,
    #     'family_size': 3,
    #     'is_parent': 'Yes'
    # }

    # ## for class 1
    # default_values = {
    #     'education': 'Basic',
    #     'income': 46344,
    #     'kidhome': 1,
    #     'teenhome': 1,
    #     'recency': 38,
    #     'wines': 11,
    #     'fruits': 1,
    #     'meat': 6,
    #     'fish': 2,
    #     'sweets': 1,
    #     'gold': 6,
    #     'num_deals_purchases': 2,
    #     'num_web_purchases': 1,
    #     'num_catalog_purchases': 2,
    #     'num_store_purchases': 5,
    #     'num_web_visits_month': 0,
    #     'accepted_cmp3': 0,
    #     'accepted_cmp4': 0,
    #     'accepted_cmp5': 0,
    #     'accepted_cmp1': 0,
    #     'accepted_cmp2': 0,
    #     'complain': 0,
    #     'response': 0,
    #     'time_enrolled_days': 299,
    #     'age': 70,
    #     'spent': 27,
    #     'living_with': 'Partner',
    #     'children': 2,
    #     'family_size': 2,
    #     'is_parent': 'Yes'
    # }

    ## for class 2
    default_values = {
        'education': 'Basic',
        'income': 58138,
        'kidhome': 0,
        'teenhome': 0,
        'recency': 58,
        'wines': 635,
        'fruits': 88,
        'meat': 546,
        'fish': 172,
        'sweets': 88,
        'gold': 88,
        'num_deals_purchases': 3,
        'num_web_purchases': 8,
        'num_catalog_purchases': 10,
        'num_store_purchases': 4,
        'num_web_visits_month': 7,
        'accepted_cmp3': 0,
        'accepted_cmp4': 0,
        'accepted_cmp5': 0,
        'accepted_cmp1': 0,
        'accepted_cmp2': 0,
        'complain': 0,
        'response': 0,
        'time_enrolled_days': 849,
        'age': 67,
        'spent': 1617,
        'living_with': 'Alone',
        'children': 0,
        'family_size': 1,
        'is_parent': 'No'
    }

    return render_template('index.html', default_values=default_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        education = request.form['education']
        income = float(request.form['income'])
        kidhome = int(request.form['kidhome'])
        teenhome = int(request.form['teenhome'])
        recency = int(request.form['recency'])
        wines = int(request.form['wines'])
        fruits = int(request.form['fruits'])
        meat = int(request.form['meat'])
        fish = int(request.form['fish'])
        sweets = int(request.form['sweets'])
        gold = int(request.form['gold'])
        num_deals_purchases = int(request.form['num_deals_purchases'])
        num_web_purchases = int(request.form['num_web_purchases'])
        num_catalog_purchases = int(request.form['num_catalog_purchases'])
        num_store_purchases = int(request.form['num_store_purchases'])
        num_web_visits_month = int(request.form['num_web_visits_month'])
        accepted_cmp3 = int(request.form['accepted_cmp3'])
        accepted_cmp4 = int(request.form['accepted_cmp4'])
        accepted_cmp5 = int(request.form['accepted_cmp5'])
        accepted_cmp1 = int(request.form['accepted_cmp1'])
        accepted_cmp2 = int(request.form['accepted_cmp2'])
        complain = int(request.form['complain'])
        response = int(request.form['response'])
        time_enrolled_days = int(request.form['time_enrolled_days'])
        age = int(request.form['age'])
        spent = int(request.form['spent'])
        living_with = request.form['living_with']
        children = int(request.form['children'])
        family_size = int(request.form['family_size'])
        is_parent = request.form['is_parent']
    
        # Map categorical features to numeric values
        education_mapping = {'Basic': 0, 'Graduation': 1, 'Master': 2, 'PhD': 3}
        living_with_mapping = {'Alone': 0, 'Partner': 1, 'Parents': 2, 'Others': 3}
        is_parent_mapping = {'Yes': 1, 'No': 0}
        
        education = education_mapping[education]
        living_with = living_with_mapping[living_with]
        is_parent = is_parent_mapping[is_parent]

        # Prepare feature array
        features = np.array([[education, income, kidhome, teenhome, recency, wines, fruits, meat, fish, sweets, gold, num_deals_purchases, num_web_purchases, num_catalog_purchases, num_store_purchases, num_web_visits_month, accepted_cmp3, accepted_cmp4, accepted_cmp5, accepted_cmp1, accepted_cmp2, complain, response, time_enrolled_days, age, spent, living_with, children, family_size, is_parent]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        clusters, target_prediction, target_prediction_proba = predict_clusters_and_target(features_scaled, kmeans, gb_model)

        # Assign cluster names
        cluster_name = ["Moderate Spenders with Minimal Family Size", "Low Income, Low Spenders with Children", "High Spenders with Small Family Size"][clusters[0]]

        return render_template('result.html', Prediction_Class=target_prediction[0], Predicted_Cluster=cluster_name, Prediction_Class_Probability=target_prediction_proba[0].round(3).tolist())

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
