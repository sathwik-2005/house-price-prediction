import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os

model = joblib.load(BASE + "/models/best_model.pkl")
scaler = joblib.load(BASE + "/models/scaler.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write("Enter house details below to get an estimated sale price.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 6000, 1500)
    garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
    year_built = st.number_input("Year Built", 1900, 2024, 2000)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

with col2:
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 500)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    yr_sold = st.selectbox("Year Sold", [2020, 2021, 2022, 2023, 2024])

st.divider()

if st.button("💰 Predict Price", use_container_width=True):
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    house_age = yr_sold - year_built
    has_garage = 1 if garage_area > 0 else 0

    sample = pd.read_csv(BASE + "/data/train.csv").drop(
        columns=['Alley','PoolQC','Fence','MiscFeature','FireplaceQu','SalePrice','Id'],
        errors='ignore'
    )
    num_cols = sample.select_dtypes(include=[np.number]).columns
    sample[num_cols] = sample[num_cols].fillna(sample[num_cols].median())
    cat_cols = sample.select_dtypes(include=['object']).columns
    for col in cat_cols:
        sample[col].fillna(sample[col].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in cat_cols:
        sample[col] = le.fit_transform(sample[col])

    sample['TotalSF'] = sample['TotalBsmtSF'] + sample['1stFlrSF'] + sample['2ndFlrSF']
    sample['HouseAge'] = sample['YrSold'] - sample['YearBuilt']
    sample['RemodAge'] = sample['YrSold'] - sample['YearRemodAdd']
    sample['HasGarage'] = (sample['GarageArea'] > 0).astype(int)

    row = sample.median().to_frame().T
    row['OverallQual'] = overall_qual
    row['GrLivArea'] = gr_liv_area
    row['GarageCars'] = garage_cars
    row['YearBuilt'] = year_built
    row['TotalBsmtSF'] = total_bsmt_sf
    row['1stFlrSF'] = first_flr_sf
    row['2ndFlrSF'] = second_flr_sf
    row['FullBath'] = full_bath
    row['GarageArea'] = garage_area
    row['TotalSF'] = total_sf
    row['HouseAge'] = house_age
    row['HasGarage'] = has_garage

    row_scaled = scaler.transform(row)
    log_price = model.predict(row_scaled)[0]
    price = np.expm1(log_price)

    st.success(f"### 💰 Estimated Sale Price: ${price:,.0f}")
    low = price * 0.90
    high = price * 1.10
    st.info(f"📊 Likely price range: **${low:,.0f}** – **${high:,.0f}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Quality", f"{overall_qual}/10")
    col2.metric("Total Area", f"{total_sf:,} sq ft")
    col3.metric("House Age", f"{house_age} years")