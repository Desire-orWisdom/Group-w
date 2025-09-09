import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("laptop_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💻 Laptop Price Predictor")

# Collect inputs
company = st.selectbox("Company", ["Dell", "HP", "Apple", "Lenovo", "Asus"])
typename = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible"])
inches = st.number_input("Inches", 10.0, 20.0, step=0.1)
ram = st.number_input("RAM (GB)", 2, 64, step=2)
cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.0, step=0.1)
ssd = st.number_input("SSD (GB)", 0, 2000, step=128)
hdd = st.number_input("HDD (GB)", 0, 2000, step=128)
flash_storage = st.number_input("Flash Storage (GB)", 0, 2000, step=128)
hybrid = st.number_input("Hybrid", 0, 2000, step=128)
cpu_series = st.selectbox("Cpu Series", ["i3", "i5", "i7", "i9"])
Gpu_brand = st.selectbox("GPU series", ['Intel', 'AMD', 'Nvidia', 'ARM'])
cpu_brand = st.selectbox("CPU brand", ['Intel', 'AMD', 'Samsung'])
Opsys = st.selectbox("Operating System", ['macOS', 'No OS', 'Mac OS X', 'Windows 10', 'Linux', 'Android',
       'Windows 10 S', 'Chrome OS', 'Windows 7'])

# inputs with one hot encoding
is_ips = st.checkbox("IPS Panel?")
is_retina = st.checkbox("Retina Display?")
is_touchscreen = st.checkbox("Touchscreen?")
is_4k = st.checkbox("4K Resolution?")
# for resolution
is_ips = int(is_ips)
is_retina = int(is_retina)
is_touchscreen = int(is_touchscreen)
is_4k = int(is_4k)

#prediction condition
if st.button("Predict Price"):
    # Make dataframe
    input_data = pd.DataFrame([{
    "Company": company,
    "TypeName": typename,
    "Inches": inches,
    "Ram": ram,
    "Cpu_speed": cpu_speed,
    "Cpu_Series" : cpu_series,
    "OpSys" : Opsys ,
    "Gpu_brand":Gpu_brand,
    "SSD": ssd,
    "HDD": hdd,
    "Flash_Storage": flash_storage,
    "Hybrid" : hybrid,
    "Is_IPS": is_ips,
    "Is_Retina": is_retina,
    "Is_Touchscreen": is_touchscreen,
    "Is_4K": is_4k ,
    "Cpu_brand" : cpu_brand,
    
        
    }])

     # ⚡ Apply same preprocessing (encode + scale)
    # Example: scale numeric cols
    num_cols = ["Inches", "Ram", "Cpu_speed", "SSD", "HDD", "Flash_Storage", "Hybrid", "Is_IPS" , "Is_Retina" , "Is_Touchscreen" , "Is_4K"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: €{prediction:.2f}")