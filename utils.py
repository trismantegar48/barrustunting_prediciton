import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Lambda
from constants import WHO_REF

@st.cache_resource
def load_resources():
    try:
        model_path = 'models/LSTM_batch8_epoch500.h5'
        scaler_path = 'models/scaler_lstm.pkl'
        data_path = 'data/datatumbuhbarru.xlsx'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, None, "File model atau scaler tidak ditemukan di folder 'models/'."

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(12, 3)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Lambda(lambda x: x[:, -6:, :])) 
        model.add(TimeDistributed(Dense(3)))
        
        model.compile(optimizer='adam', loss='mse')
        model.load_weights(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        df_real = None
        if os.path.exists(data_path):
            df_real = pd.read_excel(data_path)
            
        return model, scaler, df_real, None

    except Exception as e:
        return None, None, None, f"Terjadi kesalahan teknis: {str(e)}"

def calculate_zscore(val, age, sex, indicator):
    age = int(min(max(age, 0), 24))
    sex_key = "MALE" if sex == "Laki-laki" else "FEMALE"
    
    if indicator == 'TB':
        median = WHO_REF[f'{sex_key}_HT'][age]
        sd2 = WHO_REF[f'{sex_key}_HT_SD2'][age]
        sd_val = median - sd2
    elif indicator == 'BB':
        median = WHO_REF[f'{sex_key}_WT'][age]
        sd2 = WHO_REF[f'{sex_key}_WT_SD2'][age]
        sd_val = median - sd2
    elif indicator == 'LILA':
        median = WHO_REF['MUAC'][age]
        sd2 = WHO_REF['MUAC_SD2'][age]
        sd_val = median - sd2
    
    if sd_val == 0: return 0, median, sd2
    z = (val - median) / sd_val
    return z, median, sd2

def get_status(z):
    if z < -3: return "Gizi Buruk", "red"
    elif z < -2: return "Gizi Kurang", "orange"
    elif z < -1: return "Berisiko", "#B8860B" 
    else: return "Normal", "green"

def plot_metric_matplotlib(title, y_actual, y_pred, key_who, key_who_sd2, ylabel, usia_lalu, usia_depan, usia_saat_ini):
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(10, 5))
    
    all_plot_ages = usia_lalu + usia_depan
    all_who_ages = [min(24, a) for a in all_plot_ages]

    who_median = [WHO_REF[key_who][a] for a in all_who_ages]
    who_sd2 = [WHO_REF[key_who][a] - WHO_REF[key_who_sd2][a] for a in all_who_ages]

    ax.plot(usia_lalu, y_actual, '-o', color='blue', linewidth=2, 
            label='Aktual (12 Bln Terakhir)', markersize=6)
    
    ax.plot(usia_depan, y_pred, '-o', color='red', linewidth=2, 
            label='Prediksi (6 Bln Depan)', markersize=6)

    ax.plot(all_plot_ages, who_median, 'g-', linewidth=1.5, label='WHO Median (Ideal)', alpha=0.7)
    ax.plot(all_plot_ages, who_sd2, 'g--', linewidth=1.5, label='WHO -2SD (Batas Stunting)', alpha=0.7)

    ax.axvline(x=usia_saat_ini, color='black', linestyle=':', alpha=0.6, label='Posisi Sekarang')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Usia (Bulan)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='best', fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig