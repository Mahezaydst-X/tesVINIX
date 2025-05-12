import streamlit as st
import pandas as pd
import plotly.express as px

# --- Load Dataset ---
@st.cache_data

def load_data():
    file_path = 'UKT_Kedokteran - Sheet1.csv'  # Ensure the file is in the same folder
    data = pd.read_csv(file_path)
    return data

# --- Preprocess Data ---
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df['Universitas'] = df['Universitas'].astype(str)
    df['Program Studi'] = df['Program Studi'].astype(str)
    df['Provinsi'] = df['Provinsi'].astype(str)
    df['Daya Tampung SNBT'] = pd.to_numeric(df['Daya Tampung SNBT'], errors='coerce')
    df['Peminat SNBT'] = pd.to_numeric(df['Peminat SNBT'], errors='coerce')
    df['UKT Terendah'] = df['UKT Terendah'].str.replace('Rp', '').str.replace('.', '').astype(float)
    df['UKT Tertinggi'] = df['UKT Tertinggi'].str.replace('Rp', '').str.replace('.', '').astype(float)
    df = df.dropna(subset=['Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah', 'UKT Tertinggi'])
    return df

# --- Load and Clean Data ---
df = load_data()
df = preprocess_data(df)

# --- Sidebar Filter ---
st.sidebar.title("🔍 Filter Data")
provinsi = st.sidebar.selectbox("Pilih Provinsi", sorted(df['Provinsi'].unique()))
df_filtered_prov = df[df['Provinsi'] == provinsi]

universitas = st.sidebar.selectbox("Pilih Universitas", sorted(df_filtered_prov['Universitas'].unique()))
df_filtered_univ = df_filtered_prov[df_filtered_prov['Universitas'] == universitas]

program_studi = st.sidebar.selectbox("Pilih Program Studi", sorted(df_filtered_univ['Program Studi'].unique()))
df_filtered_prog = df_filtered_univ[df_filtered_univ['Program Studi'] == program_studi]

# --- Tabs ---
tab1, tab2 = st.tabs(["📈 Klasifikasi Peminat", "🏫 Klasifikasi Daya Tampung"])

# --- Tab 1: Peminat ---
with tab1:
    st.header("📊 Dashboard Klasifikasi Peminat")
    st.markdown("""
    🎯 **Eksplorasi Program Studi Berdasarkan Provinsi, Universitas, dan Kategori Peminat**
    """)

    kategori_peminat = st.selectbox("Pilih Kategori Peminat", ["Peminat Banyak", "Peminat Sedikit"], key="peminat")
    threshold_peminat = df['Peminat SNBT'].median()

    if kategori_peminat == "Peminat Banyak":
        result_peminat = df_filtered_prog[df_filtered_prog['Peminat SNBT'] > threshold_peminat]
    else:
        result_peminat = df_filtered_prog[df_filtered_prog['Peminat SNBT'] <= threshold_peminat]

    if result_peminat.empty:
        st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")
    else:
        st.success(f"Menampilkan {len(result_peminat)} program studi yang sesuai dengan filter Anda.")
        st.dataframe(result_peminat)

        fig = px.bar(result_peminat, x="Program Studi", y="Peminat SNBT", title="Jumlah Peminat Program Studi")
        st.plotly_chart(fig)

# --- Tab 2: Daya Tampung ---
with tab2:
    st.header("🏫 Dashboard Klasifikasi Daya Tampung")
    st.markdown("""
    🎯 **Eksplorasi Program Studi Berdasarkan Provinsi, Universitas, dan Kategori Daya Tampung**
    """)

    kategori_daya = st.selectbox("Pilih Kategori Daya Tampung", ["Kuota Besar", "Kuota Kecil"], key="daya")
    threshold_daya = df['Daya Tampung SNBT'].median()

    if kategori_daya == "Kuota Besar":
        result_daya = df_filtered_prog[df_filtered_prog['Daya Tampung SNBT'] > threshold_daya]
    else:
        result_daya = df_filtered_prog[df_filtered_prog['Daya Tampung SNBT'] <= threshold_daya]

    if result_daya.empty:
        st.error("Tidak ada program studi yang memenuhi kriteria.")
    else:
        st.success(f"✅ Menampilkan program studi di {universitas}, {provinsi} dengan kategori {kategori_daya} (Daya Tampung):")
        st.dataframe(result_daya)

        fig2 = px.bar(result_daya, x="Program Studi", y="Daya Tampung SNBT", title="Daya Tampung Program Studi")
        st.plotly_chart(fig2)
