import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib




# Load the pre-trained models

## Classification
xgb_classification_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Classification\xgboost_model.pkl')
rf_classification_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Classification\random_forest.pkl')
dt_classification_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Classification\Decision_tree.pkl')
svc_kernal_rbf_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Classification\SVC_kernal_RBF.pkl')
neural_network_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Classification\neural_network.pkl')

## Regression
xgb_prediction_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\xgboost\xgboost_model_all_features.pkl')
gdbt_prediction_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\gbdt\gbdt_model_all_features.pkl')
ridge_prediction_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\ridge\ridge_model.pkl')
svr_prediction_model = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\SVR\best_svr.pkl')

# Pre data for classification

## Load and prepare the data
df = pd.read_json(r'C:\Code\ML_Project\Data\preprocessed_movieDB.json')
df = df[df['revenue'] < 100 * 1e+6]

## Creating binning for classification
bin_size = 20
df['revenue_bin'] = np.floor(df['revenue'] / (bin_size * 1e+6)).astype(int)

## MinMaxScaler for budget
scaler1 = MinMaxScaler()
df['budget'] = scaler1.fit_transform(df['budget'].values.reshape(-1, 1))

## Create feature and class target
X1 = df.drop(['title', 'revenue', 'revenue_bin'], axis=1)
y1= df['revenue_bin']

## Standard Scaler
scaler1 = StandardScaler()
X1= scaler1.fit_transform(X1)

# Pre data for regression
## Load and prepare the data
df1 =  pd.read_json(r'C:\Code\ML_Project\Data\preprocessed_movieDB.json')
X2 = df1.drop(['revenue','title'], axis=1)
cols =['budget']
for col in cols:
    df1[col] = np.log(df1[col] + 1)
y2 = df1['revenue']


# Streamlit app
st.title('Movie Revenue Prediction')

# Prediction type selection
prediction_type = st.radio('Choose Prediction Type:', ('Classification', 'Regression'))

if prediction_type == 'Classification':
    st.header('Movie Revenue Classification')

    # Algorithm selection
    class_algorithm = st.selectbox('Select Classification Algorithm:', ('XGBoost','Decision Tree', 'Random Forest', 'SVC','Neural Network'))
    if class_algorithm == 'XGBoost':
        classification_model = xgb_classification_model
    elif class_algorithm == 'Random Forest':
        classification_model = rf_classification_model
    elif class_algorithm == 'Decision Tree':
        classification_model = dt_classification_model
    elif class_algorithm == 'SVC':
        classification_model = svc_kernal_rbf_model
    else:
        classification_model = neural_network_model 

    #Row selection
    selected_index = st.selectbox('Select a data row index for classification', df.index)

    if st.button('Classify Selected Row'):
        selected_data = X1[selected_index].reshape(1, -1)
        prediction = classification_model.predict(selected_data)
        if classification_model==neural_network_model:
            prediction = np.argmax(prediction, axis=1)
        st.write(f'Predicted Revenue Bin: {prediction[0]}')
        st.write('Data for Selected Row:')
        st.write(df.iloc[selected_index].to_frame().T)

    st.write('---')


elif prediction_type == 'Regression':   

    st.header('Movie Revenue Regression')

    # Algorithm selection
    class_algorithm = st.selectbox('Select Classification Algorithm:', ('XGBoost','GDBT','Ridge','SVR', 'Random Forest'))
    if class_algorithm == 'XGBoost':
        regression_model = xgb_prediction_model
    elif class_algorithm =='GDBT':
        regression_model = gdbt_prediction_model
    elif class_algorithm == 'Ridge':
        regression_model = ridge_prediction_model
    elif class_algorithm == 'SVR':
        regression_model = svr_prediction_model

    # Option to select a row from the dataset
    selected_index = st.selectbox('Select a data row index for regression', df1.index)

    if st.button('Predict Revenue for Selected Row'):
        if class_algorithm == 'XGBoost':
            scaler2 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\xgboost\scaler_xgboost_all_features.pkl')
        elif class_algorithm == 'GDBT':
            scaler2 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\gbdt\scaler_gbdt_all_features.pkl')
        elif class_algorithm == 'Ridge':
            scaler2 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\ridge\scaler1.pkl')
        elif class_algorithm == 'SVR':
            scaler2 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\SVR\scaler1.pkl')

        X2=scaler2.transform(X2)
        selected_data = X2[selected_index].reshape(1, -1)
        prediction = regression_model.predict(selected_data)
        if class_algorithm == 'Ridge':
            scaler3 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\ridge\scaler2.pkl')
            st.write(f'Predicted Revenue: ${scaler3.inverse_transform(prediction.reshape(-1, 1))[0][0]:,.2f}')
        elif class_algorithm == 'SVR':
            scaler3 = joblib.load(r'C:\Code\IT3190E_Group_32\Source Code\Model\Regression\SVR\scaler2.pkl')
            st.write(f'Predicted Revenue: ${scaler3.inverse_transform(prediction.reshape(-1, 1))[0][0]:,.2f}')
        else:
            prediction = np.exp(prediction)
            st.write(f'Predicted Revenue: ${float(prediction[0]):,.2f}')
        st.write('Data for Selected Row:')
        st.write(df1.iloc[selected_index].to_frame().T)

    st.write('---')
