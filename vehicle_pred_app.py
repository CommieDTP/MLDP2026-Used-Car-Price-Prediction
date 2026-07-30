import joblib
import streamlit as st
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

# Background Styling
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    .subheader {
            text-align: center;
            color: #FFFFFF;
            font-weight : medium;
            font-size : 4.0rem;
            font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """,
    ## Parameter to allow for custom CSS Styling, Custom layout elements, and embedding external media that streamlit doesnt support natively
    unsafe_allow_html=True 
)

# Load PKL asset files
@st.cache_resource ## Cache the data from the pkl files as loading large models everytime a slider is moved incurs delays & latency
def load_pkl():
    gbr_model = joblib.load("UK_Car_Prices_FINAL_rs_gbr_v3.pkl")
    r_scaler = joblib.load("RScaler_FINAL_gbr_v3.pkl")
    feature_columns = joblib.load("feature_columns_FINAL_v3.pkl")
    return gbr_model, r_scaler, feature_columns

gbr_model, r_scaler, feature_columns = load_pkl()


# Categorical columns lists for all selected features

## Nested Dictionary to store all possible brand-model pairs for all samples
brand_model_dict = {'Abarth': ['500', 'Others'], 'Alfa': ['Others'], 'Audi': ['A4', 'A3', 'Others', 'A1', 'A5 DIESEL COUPE'], 'BMW': ['3 Series', '1 Series', '5 Series', 'Others', 'X1'], 'Chevrolet': ['Others'], 'Chrysler': ['Others'], 'Citroen': ['Others', 'C1', 'C3'], 'DS': ['Others'], 'Dacia': ['Others'], 'Daewoo': ['Others'], 'Daihatsu': ['Others'], 'Fiat': ['500', 'Others', 'Punto EVO'], 'Ford': ['Mondeo', 'KA', 'Focus', 'Others', 'Fiesta', 'Kuga', 'Ecosport'], 'Honda': ['Accord', 'Civic', 'Jazz', 'Others', 'Cr-v'], 'Hyundai': ['Others', 'i30', 'i20', 'i10', 'Tucson'], 'Infiniti': ['Others'], 'Jaguar': ['XE', 'Others', 'XF'], 'Jeep': ['Others'], 'KIA': ['RIO', 'Ceed', 'Picanto', 'Others', 'Sportage'], 'Land': ['Others', 'Rover Freelander 2'], 'Lexus': ['Others'], 'MG': ['Others'], 'MINI': ['Hatch', 'Others'], 'Marcos': ['Others'], 'Maserati': ['Others'], 'Mazda': ['Mazda2', 'Mazda3', 'Others'], 'Mercedes-Benz': ['A Class', 'Others', 'C Class', 'E Class'], 'Mitsubishi': ['Outlander', 'Others'], 'Nissan': ['Others', 'Note', 'Micra', 'Qashqai', 'Qashqai+2', 'Juke'], 'Peugeot': ['207', '308 SW', 'Others', '208', '2008', '107', '308', '407', '3008'], 'Porsche': ['Others'], 'Proton': ['Others'], 'Renault': ['Others', 'Clio', 'Megane', 'Grand Scenic', 'Captur'], 'Rover': ['Others'], 'SEAT': ['Ibiza', 'Others', 'Leon'], 'SKODA': ['Fabia', 'Others', 'Octavia'], 'Saab': ['Others'], 'Smart': ['Others'], 'Ssangyong': ['Others'], 'Subaru': ['Impreza', 'Others'], 'Suzuki': ['Others', 'Swift'], 'Toyota': ['Auris', 'Yaris', 'Corolla', 'Others', 'Verso', 'Aygo'], 'Vauxhall': ['Corsa', 'Insignia', 'Astra', 'Meriva', 'Zafira', 'Others', 'Mokka', 'Astra GTC', 'Adam', 'Crossland X'], 'Volkswagen': ['Polo', 'Beetle', 'Golf', 'Scirocco', 'Passat', 'Others', 'Tiguan', 'T-cross'], 'Volvo': ['Others', 'V70']}
# sorted - All numerical & text columns are sorted in chronological/alphabetical/numerical order
brands = sorted(list(brand_model_dict.keys()))
engines = sorted([1.4, 1.2, 2.0, 1.6, 1.8, 1.3, 1.9, 2.4, 1.5, 2.2, 1.0, 3.0, 3.5, 2.5, 2.1, 2.8, 1.1, 5.0, 0.8, 2.7, 1.7, 4.3, 3.2, 2.3, 3.3, 0.9, 4.4, 3.7, 5.5, 4.2, 6.3])
gearboxes = sorted(['Manual', 'Automatic'])
body_types = sorted(['Hatchback', 'Coupe', 'Saloon', 'Estate', 'Convertible', 'MPV', 'SUV', 'Others', 'Pickup'])
fuel_types = sorted(['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'])
emission_classes = sorted(['Euro 6', 'Euro 4', 'Euro 5', 'Euro 3', 'Unknown', 'Euro 2', 'Euro 1'])

# APP LAYOUT & UI

# Headers Title Text
st.image("2-cecb3abb-crop.JPG", use_container_width=True)
st.markdown("<h1 class='main-header'>Used Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Unsure about the price of your vehicle?</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill in the details about the vehicle below & we'll give you an estimate!</p>", unsafe_allow_html=True)
st.markdown("---")

# Create a sleek two-column layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Vehicle Details")
    
    # Registration year slider
    registration_year = st.slider("Registration Year", min_value=2000, max_value=2024, value=2012)
    # Text(Number) Box to select Mileage clocked with manual selectors in increment of 1000 miles
    mileage = st.number_input("Total Mileage (miles)", min_value=0, max_value=300000, value=50000, step=1000)
    
    # Categorical Inputs (Arranged in smaller columns for a compact look)
    c1, c2 = st.columns(2)
    with c1:
        # Left Column - Dropdown Boxes
        brand = st.selectbox("Brand", brands)
        engine = st.selectbox("Engine Capacity (L)", engines)
        body_type = st.selectbox("Body Type", body_types)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        
    with c2:
        # Right Column - Dropdown Boxes
        ## Dynamically fetch the models for that specific brand using dictionary keys
        ## Empty list [] prevents crashes if a brand has no models)
        available_models = sorted(brand_model_dict.get(brand, []))
        # Only shows valid models from each car brand (e.g. BMW - 3 series, 5 series)
        model_name = st.selectbox("Model", available_models)
        
        gearbox = st.selectbox("Transmission", gearboxes)
        emission_class = st.selectbox("Emission Class (Euro)", emission_classes)
with col2:
    st.subheader("Estimated Value (£)")
    st.warning("NOTE : This prediction is an only an estimate based on the displayed circumstances. Actual resale prices may vary.")
    st.info("Model : If your vehicle model is not listed, please select 'Others' as your model.\n\n" \
            "Body Type : Select the general category of the Body Type of the vehicle. If you're unsure, select 'Others'.\n"
            )
    
    st.write("") 
    st.write("")
    
    # Predict buttons
    if st.button("Predict Price (£)", type="primary", use_container_width=True):
        
        # Implemented Feature Engineering logic
        ## Calculate vehicle age by subtracting current year by year of registration
        vehicle_age = 2024 - registration_year
        if vehicle_age <= 0:
            vehicle_age = 1  # Prevent divide by 0 errors. If the car is brand new (age = 0) list the car as 1 year old
        ## Mileage travelled on average in a year by vehicle age
        mileage_per_year = mileage / vehicle_age
        
        df_input = pd.DataFrame({
            'Registration_Year': [registration_year],
            'Mileage(miles)': [mileage],
            'Engine': [engine],
            'Brand': [brand],
            'Gearbox': [gearbox],
            'Body type': [body_type],
            'Emission Class': [emission_class],
            'Fuel type': [fuel_type],
            'Model': [model_name],
            'Vehicle_Age': [vehicle_age],
            'Mileage_Per_Year': [mileage_per_year]
        })
        
        # OHE - based on dataframe input
        df_input = pd.get_dummies(df_input)
        
        # Align columns with training data (Fills missing categories with 0)
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        
        # Scale the data using RobustScaler (e.g. high mileage outliers)
        df_input_scaled = r_scaler.transform(df_input)
        
        # Predict & Display
        prediction = gbr_model.predict(df_input_scaled)[0]
        
        st.success("Valuation Complete!")
        st.metric(label="Estimated Resale Price", value=f"£{prediction:,.2f}")