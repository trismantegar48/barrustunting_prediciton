import streamlit as st
import pandas as pd
import numpy as np
import utils 
from constants import CUSTOM_CSS

st.set_page_config(
    page_title="Sistem Prediksi Stunting - Barru",
    layout="wide"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# LOAD RESOURCES
model, scaler, df_real, err_msg = utils.load_resources()


st.sidebar.title("Input Data Balita")

if err_msg:
    st.error(err_msg)

input_mode = "Input Manual"
selected_child_id = None

if df_real is not None:
    use_dataset = st.sidebar.checkbox("Pilih dari Data Excel?", value=True)
    
    if use_dataset:
        child_counts = df_real['ID_Anak'].value_counts()
        valid_ids = child_counts[child_counts >= 12].index.tolist()
        
        st.sidebar.success(f"Data Terhubung: {len(valid_ids)} Anak (Min. 12 bulan data)")
        
        input_mode = "Dataset"
        child_options = {id_anak: f"{id_anak} - {df_real[df_real['ID_Anak']==id_anak]['Nama'].iloc[0]}" for id_anak in valid_ids}
        
        selected_child_id = st.sidebar.selectbox(
            "Pilih ID Anak:", 
            options=list(child_options.keys()),
            format_func=lambda x: child_options[x]
        )

# DATA DEFAULT 
default_nama = "Anak A"
default_jk = "Laki-laki"
default_usia = 12
initial_data = {
    "Bulan_Ke": list(range(1, 13)),
    "Tinggi (cm)": [0.0]*12,
    "Berat (kg)": [0.0]*12,
    "LiLA (cm)": [0.0]*12
}

if input_mode == "Dataset" and selected_child_id:
    child_data = df_real[df_real['ID_Anak'] == selected_child_id].sort_values('Usia Saat Ukur')
    default_nama = child_data['Nama'].iloc[-1]

    raw_jk = child_data['JK'].iloc[-1]
    if str(raw_jk).upper() in ['L', 'LAKI-LAKI', 'MALE']:
        default_jk = "Laki-laki"
    else:
        default_jk = "Perempuan"

    default_usia = int(child_data['Usia Saat Ukur'].iloc[-1])
    
    if len(child_data) >= 12:
        last_12 = child_data.tail(12)
        initial_data["Tinggi (cm)"] = last_12['Tinggi'].values.tolist()
        initial_data["Berat (kg)"] = last_12['Berat'].values.tolist()
        initial_data["LiLA (cm)"] = last_12['LiLA'].values.tolist()
    else:
        st.sidebar.warning("Data anak ini kurang dari 12 bulan.")

# FORM INPUT
nama_anak = st.sidebar.text_input("Nama Anak", default_nama)
jk_index = 0 if default_jk == "Laki-laki" else 1
jk = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=jk_index)
usia_saat_ini = st.sidebar.number_input("Usia Terakhir (Bulan)", value=default_usia, min_value=12)

st.sidebar.markdown("### Data 12 Bulan Terakhir (Input/Edit)")
df_input_template = pd.DataFrame(initial_data)

edited_df = st.sidebar.data_editor(
    df_input_template,
    column_config={
        "Bulan_Ke": st.column_config.NumberColumn("Urutan (Bln ke-)", disabled=True),
        "Tinggi (cm)": st.column_config.NumberColumn(required=True, min_value=0.0, max_value=150.0, format="%.1f"),
        "Berat (kg)": st.column_config.NumberColumn(required=True, min_value=0.0, max_value=50.0, format="%.1f"),
        "LiLA (cm)": st.column_config.NumberColumn(required=True, min_value=0.0, max_value=30.0, format="%.1f"),
    },
    hide_index=True,
    num_rows="fixed"
)

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("Prediksi Risiko Stunting", type="primary", use_container_width=True)

st.markdown("<div class='main-header'>Dashboard Prediksi Risiko Stunting Kabupaten Barru</div>", unsafe_allow_html=True)
st.markdown(f"""
    <div style="text-align: center; font-size: 18px; color: #white; margin-bottom: 20px; margin-top: -10px;">
        <span style="margin-right: 15px;"><b>Nama :</b> {nama_anak}</span> |
        <span style="margin: 0 15px;"><b>Jenis Kelamin :</b> {jk}</span> |
        <span style="margin-left: 15px;"><b>Usia Saat Ini :</b> {usia_saat_ini} Bulan</span>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Bagian Info Aplikasi
if not btn_predict:
    with st.expander("ℹ️ Tentang Aplikasi (Tujuan & Manfaat)", expanded=True):
        st.markdown("""
        **Tujuan Aplikasi :**
        Aplikasi ini dikembangkan untuk memprediksi risiko stunting pada balita di Kabupaten Barru menggunakan metode *Deep Learning* (LSTM). Tujuannya adalah mengimplementasikan hasil prediksi model LSTM ke dalam *dashboard* yang dapat memvisualisasikan status risiko stunting berdasarkan standar WHO.
        
        **Manfaat Aplikasi :**
        1.  Sebagai alat deteksi dini untuk mengatasi peningkatan prevalensi stunting.
        2.  Membantu memantau tren pertumbuhan anak serta meningkatkan pemahaman akan pentingnya pemantauan secara rutin sebagai upaya pencegahan stunting.
        3.  Menyajikan gambaran status gizi masa depan balita secara akurat untuk mendukung pengambilan langkah pencegahan yang lebih cepat dan tepat.
        """)

if err_msg:
    st.error(f"Error Loading Model: {err_msg}")
    st.warning("Pastikan file model (.h5) dan scaler (.pkl) ada di folder 'models/'.")

elif btn_predict:
    with st.spinner('Sedang memproses prediksi...'):
        try:
            # 1. Preprocessing Input
            input_raw = edited_df[["Tinggi (cm)", "Berat (kg)", "LiLA (cm)"]].values
            input_norm = scaler.transform(input_raw) 
            input_seq = input_norm.reshape(1, 12, 3)
            
            # 2. Prediksi
            pred_seq = model.predict(input_seq, verbose=0) 
            
            # 3. Inverse Scaling
            pred_real = scaler.inverse_transform(pred_seq.reshape(-1, 3))
            
            # HASIL & VISUALISASI
            usia_lalu = [usia_saat_ini - 11 + i for i in range(12)] 
            usia_depan = [usia_saat_ini + i + 1 for i in range(6)]
            
            df_pred = pd.DataFrame(pred_real, columns=["Tinggi", "Berat", "LiLA"])
            df_pred["Usia_Prediksi"] = usia_depan
            
            status_list = []
            for idx, row in df_pred.iterrows():
                # Hitung Z-score pakai fungsi dari utils.py
                z_tb, median_tb, _ = utils.calculate_zscore(row['Tinggi'], row['Usia_Prediksi'], jk, 'TB')
                stat_tb, color_tb = utils.get_status(z_tb)

                z_bb, median_bb, _ = utils.calculate_zscore(row['Berat'], row['Usia_Prediksi'], jk, 'BB')
                stat_bb, color_bb = utils.get_status(z_bb)

                z_lila, median_lila, _ = utils.calculate_zscore(row['LiLA'], row['Usia_Prediksi'], jk, 'LILA')
                stat_lila, color_lila = utils.get_status(z_lila)
                
                status_list.append({
                    "Bulan Ke": idx + 1,
                    "Usia (Bln)": int(row['Usia_Prediksi']),
                    "Tinggi (cm)": row['Tinggi'], "Status TB": stat_tb, "Color_TB": color_tb, "Target TB": median_tb,
                    "Berat (kg)": row['Berat'], "Status BB": stat_bb, "Color_BB": color_bb, "Target BB": median_bb,
                    "LiLA (cm)": row['LiLA'], "Status LiLA": stat_lila, "Color_LiLA": color_lila, "Target LiLA": median_lila
                })
            
            df_status = pd.DataFrame(status_list)
            last_stat = df_status.iloc[-1]
            st.success("✅ Prediksi Berhasil!")
            
            col1, col2, col3 = st.columns(3)

            def create_metric_card(col, title, value, unit, status, color, target_val):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 5px solid {color};">
                        <h4>{title} (6 Bln ke depan)</h4>
                        <h2>{value:.2f} <small>{unit}</small></h2>
                        <span style="color:{color}; font-weight:bold; font-size:18px;">{status}</span><br>
                        <div class="metric-ref" style="font-size:12px; margin-top:5px; color:#ccc;">Standar Normal : {target_val:.1f} {unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

            create_metric_card(col1, "Prediksi TB", last_stat['Tinggi (cm)'], "cm", last_stat['Status TB'], last_stat['Color_TB'], last_stat['Target TB'])
            create_metric_card(col2, "Prediksi BB", last_stat['Berat (kg)'], "kg", last_stat['Status BB'], last_stat['Color_BB'], last_stat['Target BB'])
            create_metric_card(col3, "Prediksi LiLA", last_stat['LiLA (cm)'], "cm", last_stat['Status LiLA'], last_stat['Color_LiLA'], last_stat['Target LiLA'])
            
            st.markdown("---")

            # VISUALISASI
            tab1, tab2 = st.tabs(["📈 Grafik Pertumbuhan", "📋 Detail Data"])
            sex_key = "MALE" if jk == "Laki-laki" else "FEMALE"
            
            with tab1:
                # Memanggil fungsi plot dari utils.py
                st.markdown("##### 1. Grafik Tinggi Badan")
                st.pyplot(utils.plot_metric_matplotlib(f"Trajektori Tinggi Badan: {nama_anak} ({jk})", edited_df["Tinggi (cm)"], df_pred["Tinggi"], f'{sex_key}_HT', f'{sex_key}_HT_SD2', "Tinggi (cm)", usia_lalu, usia_depan, usia_saat_ini))

                st.markdown("##### 2. Grafik Berat Badan")
                st.pyplot(utils.plot_metric_matplotlib(f"Trajektori Berat Badan: {nama_anak} ({jk})", edited_df["Berat (kg)"], df_pred["Berat"], f'{sex_key}_WT', f'{sex_key}_WT_SD2', "Berat (kg)", usia_lalu, usia_depan, usia_saat_ini))

                st.markdown("##### 3. Grafik Lingkar Lengan Atas (LiLA)")
                st.pyplot(utils.plot_metric_matplotlib(f"Trajektori LiLA: {nama_anak} ({jk})", edited_df["LiLA (cm)"], df_pred["LiLA"], 'MUAC', 'MUAC_SD2', "LiLA (cm)", usia_lalu, usia_depan, usia_saat_ini))

            with tab2:
                st.markdown("### Detail Prediksi Per Bulan")
                def color_status_cell(val):
                    color = 'black'
                    if 'Buruk' in val: color = 'red'
                    elif 'Kurang' in val or 'Stunted' in val: color = 'orange'
                    elif 'Berisiko' in val: color = '#B8860B'
                    elif 'Normal' in val: color = 'green'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(df_status[["Usia (Bln)", "Tinggi (cm)", "Status TB", "Berat (kg)", "Status BB", "LiLA (cm)", "Status LiLA"]].style.applymap(color_status_cell, subset=['Status TB', 'Status BB', 'Status LiLA']).format("{:.2f}", subset=["Tinggi (cm)", "Berat (kg)", "LiLA (cm)"]), use_container_width=True)
                st.markdown("---")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

else:
    st.info("↞ Silakan cek data pertumbuhan anak pada panel sebelah kiri, lalu tekan tombol **'Prediksi Risiko Stunting'**.")