import streamlit as st
import numpy as np
import joblib

st.title('BoostHome-House Price Predictor Tool')

new_df=joblib.load('new_df.pkl')
df=joblib.load('df.pkl')
model=joblib.load('Booster.save_model.pkl')

CityName=st.selectbox('City-Name',new_df['city'].unique())
CityPin=st.selectbox('Pin-Code',df['city'].unique())
bedrooms=st.selectbox('bedrooms',df['bedrooms'].unique())
bathrooms=st.selectbox('bathrooms',df['bathrooms'].unique())
sqftliving=st.selectbox('Sqft-living',df['sqft_living'].unique())
sqftabove=st.selectbox('Sqft-Above',df['sqft_above'].unique())
sqftbasement=st.selectbox('Sqft-Basement',df['sqft_basement'].unique())
sqftlot=st.selectbox('Sqft-lot',df['sqft_lot'].unique())
floors=st.selectbox('No of Floors',df['floors'].unique())
waterfront=st.selectbox('Waterfront',['Yes','No'])
condition=st.selectbox('Condition of the house',df['condition'].unique())



if st.button('Predcit-Price'):
    if waterfront=='Yes':
        waterfront=1
    else:
        waterfront=0
    
    q=[]
    q=np.array([bedrooms,bathrooms,sqftliving,sqftlot,floors,waterfront,condition,sqftabove,sqftbasement,CityPin])
    q=q.reshape(1,10)
    
    st.title(model.predict(q))




