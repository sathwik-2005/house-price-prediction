import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/sathwik-2005/house-price-prediction/main/data/train.csv")
    df.drop(columns=['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'], inplace=True, errors='ignore')
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df['SalePrice'] = np.log1p(df['SalePrice'])
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    X = df.drop(columns=['SalePrice','Id'])
    y = df['SalePrice']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X

model, scaler, X_ref = train_model()

st.title("🏠 House Price Predictor")
st.write("Enter house details below to get an estimated sale price.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 6000, 1500)
    garage_cars = st.selectbox("Garage Capacity (cars)", [0,1,2,3,4])
    year_built = st.number_input("Year Built", 1900, 2024, 2000)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

with col2:
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 500)
    full_bath = st.selectbox("Full Bathrooms", [1,2,3,4])
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    yr_sold = st.selectbox("Year Sold", [2020,2021,2022,2023,2024])

st.divider()

if st.button("💰 Predict Price", use_container_width=True):
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    house_age = yr_sold - year_built
    has_garage = 1 if garage_area > 0 else 0

    row = X_ref.median().to_frame().T
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
    st.info(f"📊 Likely price range: **${price*0.90:,.0f}** – **${price*1.10:,.0f}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Quality", f"{overall_qual}/10")
    c2.metric("Total Area", f"{total_sf:,} sq ft")
    c3.metric("House Age", f"{house_age} years")