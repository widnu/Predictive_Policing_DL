# https://stackoverflow.com/questions/17917254/how-to-install-flask-on-windows
# https://stackoverflow.com/questions/40963401/flask-dynamic-data-update-without-reload-page

# Run the following command in cmd:
#   set FLASK_APP=nz_flask_app.py
#   flask run

# In Browser:
#   http://127.0.0.1:5000/

import numpy as np
import pandas as pd
from flask import Flask
from flask import request, redirect, render_template

import folium

from joblib import dump, load
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# App config.
DEBUG = True
app = Flask(__name__)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

def get_area():
    # read crime data
    area_csv = '../../data/master/nz_area.csv'
    df_area = pd.read_csv(area_csv, encoding='utf-8-sig')
    return df_area

def build_crime_factor(request):
    print("start build_crime_factor...")
    
    req = request.form
    day = req.get('day')
    month = req.get('month')
    hour_partition = req.get('hour_partition')
    area_0 = req.get('area_0')
    area_1 = req.get('area_1')
    crime_type = req.get('crime_type')
        
    print(day)
    print(month)
    print(hour_partition)
    print(area_0)
    print(area_1)
    print(crime_type)
    
    feature_var = ['MONTH', 'DAY', 'HOUR_PARTITION', 'AREA_0', 'AREA_1', 'CRIME_TYPE', '3_DAY_AREA_CRIME_MEAN']
    
    ##################################################
    df_area = get_area()
    df_area = df_area[df_area['AREA_1'] == area_1]
    df_area = df_area['AREA_0']
    df_area = df_area.drop_duplicates().to_frame('AREA_0')
    
    crime_factors = []
    for index, row in df_area.iterrows():
        values = [int(month), int(day), int(hour_partition), row['AREA_0'], area_1, crime_type, 5]
        zipped = zip(feature_var, values)
        a_dictionary = dict(zipped)
        print(a_dictionary)
        crime_factors.append(a_dictionary)
    
    df_crime = pd.DataFrame(crime_factors, columns=feature_var)
    # df_crime.info()
    df_crime.describe()
    
    return df_crime
    
def process_predict(df_crime_obj, algorithm):
    print("start process_predict...")
    
    X = df_crime_obj
    print(X)
    
    encoder_path = '../../output/models/encoder/'
    oe = load(encoder_path + 'nz_OrdinalEncoder.joblib')
    print(oe.categories_)
    X = oe.transform(X)
    
    # select the algorithm
    model_file_name = 'NewZealandRandomForestClassifier_depth_20.joblib'
    
    if algorithm == 'Logistic Regression':
        model_file_name = 'NewZealandLogisticRegression.joblib'
    elif algorithm == 'Naive Bayes':
        model_file_name = 'NewZealandGaussianNB.joblib'
    elif algorithm == 'K-Nearest Neighbors':
        model_file_name = 'NewZealandKNeighborsClassifier_k12.joblib'
    elif algorithm == 'Decision Tree':
        model_file_name = 'NewZealandDecisionTreeClassifier_depth_20.joblib'
    elif algorithm == 'XG Boost':
        model_file_name = 'NewZealandXGBClassifier_depth_12.joblib'
    elif algorithm == 'Random Forrest':
        model_file_name = 'NewZealandRandomForestClassifier_depth_20.joblib'
        
    model_path = '../../output/models/set_2_no_pca/'
    clf = load(model_path + model_file_name) 
    y_pred = clf.predict(X)
    print(y_pred)
    return y_pred

@app.route('/predict', methods=['POST'])
def request_predict():    
    df_crime = build_crime_factor(request)
    
    algorithm = request.form.get('algorithm')
    print(algorithm)
    
    y_pred = process_predict(df_crime, algorithm)
    df_crime['RISK'] = y_pred.tolist()
    
    df_area = get_area()
    df_area = df_area.groupby(['AREA_0', 'AREA_1']).first()
    df_crime = pd.merge(df_crime, df_area, on=['AREA_0', 'AREA_1'], how='inner')
    
    map_html = get_map(df_crime)
    
    return map_html

@app.route('/get_map')
def get_map(df_crime_obj):
    nz_lat_lon = (-36.848461, 174.8860)
    map = folium.Map(location=nz_lat_lon, default_zoom_start=5)
    
    for index, row in df_crime_obj.iterrows():
        popup_msg = row['AREA_1'] + "<br>"
        popup_msg += row['AREA_0'] + "<br>"
        popup_msg += "Crime Risk: " + str(row['RISK'])
        
        if row['RISK'] == 0:
            mark_color = 'green'
        elif row['RISK'] == 1:
            mark_color = 'blue'
        elif row['RISK'] == 2:
            mark_color = 'red'
            
        folium.Marker((row['LATITUDE'], row['LONGITUDE']), popup=popup_msg, icon=folium.Icon(color=mark_color)).add_to(map)
        
    return map._repr_html_()

@app.route('/get_area_unit_json')
def get_area_unit_json():
    print("Load Area Unit...")
    area_1 = request.args.get('area_1')
    df_area = get_area()
    df_area = df_area[df_area['AREA_1'] == area_1]
    df_area = df_area['AREA_0']
    df_area = df_area.drop_duplicates()
    return df_area.to_json(orient='values')

@app.route('/get_territorial_auth_json')
def get_territorial_auth_json():
    print("Load Terriorial Authority...")
    df_area = get_area()
    df_area = df_area['AREA_1']
    df_area = df_area.drop_duplicates()
    return df_area.to_json(orient='values')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()