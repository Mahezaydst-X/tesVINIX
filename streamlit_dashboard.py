import streamlit as st
import pandas as pd
# import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


# --- Load Dataset ---
@st.cache_data
def load_data():
    file_path = 'UKT_Kedokteran - Sheet1.csv'  # Ensure the file is in the same folder
    data = pd.read_csv(file_path)
    return data

# --- Preprocess Data ---
def preprocess_data(df):
    # Clean up the column names
    df.columns = df.columns.str.strip()
    
    # Convert relevant columns to appropriate data types
    df['Universitas'] = df['Universitas'].astype(str)
    df['Program Studi'] = df['Program Studi'].astype(str)
    df['Provinsi'] = df['Provinsi'].astype(str)
    
    # Convert to numeric, if errors occur, set those to NaN
    df['Daya Tampung SNBT'] = pd.to_numeric(df['Daya Tampung SNBT'], errors='coerce')
    df['Peminat SNBT'] = pd.to_numeric(df['Peminat SNBT'], errors='coerce')
    
    # Handle 'UKT Terendah' and 'UKT Tertinggi' columns
    df['UKT Terendah'] = df['UKT Terendah'].str.replace('Rp', '').str.replace('.', '').astype(float)
    df['UKT Tertinggi'] = df['UKT Tertinggi'].str.replace('Rp', '').str.replace('.', '').astype(float)
    
    # Drop rows with missing values (if any)
    df = df.dropna(subset=['Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah', 'UKT Tertinggi'])
    return df

# --- Streamlit Title ---
st.title("📊 Dashboard UKT Fakultas Kedokteran")


# Insight dan Tim
st.markdown("""
<div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:25px; border-left: 5px solid #0066cc;">
    <h3 style="color:#0066cc;">📌 Insight 1: Perguruan Tinggi Negeri dan Swasta di Indonesia yang memiliki fakultas Kedokteran dan daya tampungnya di tahun 2025/2026 dan berapa besarnya UKT dan biaya lainnya</h3>
    <p style="font-size:16px;">Program studi dengan kuota besar dan peminat sedikit cenderung menawarkan peluang diterima yang lebih tinggi. Sementara itu, program studi dengan UKT rendah memberikan keuntungan bagi mahasiswa yang mencari biaya kuliah lebih terjangkau.</p>
    <hr style="border:1px dashed #ccc;">
    <p style="font-size:14px;"><strong>Disusun oleh Tim:</strong><br>
    I Gusti Ngurah Bagus Lanang Purbhawa - Universitas Udayana<br>
    I Kadek Agus Candra Widnyana (534) - Universitas Udayana<br>
    I Kadek Angga Kusuma Diatmika (535)- Universitas Udayana<br>
    I Komang Maheza Yudistia (536) - Universitas Udayana
    </p>
</div>
""", unsafe_allow_html=True)


# Load and preprocess the data
df = load_data()
df = preprocess_data(df)

# --- Initial Overview Visualizations ---
st.header("📊 Data Overview")

# 1. **Total Peminat per Program (Overall)**
st.subheader("📈 Total Peminat per Program (Overall)")
fig1 = px.bar(
    df,
    x='Program Studi', y='Peminat SNBT',
    title="Total Peminat per Program",
    labels={'Program Studi': 'Program Studi', 'Peminat SNBT': 'Jumlah Peminat'},
    color='Peminat SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig1)

# 2. **Total Daya Tampung per Program (Overall)**
st.subheader("📈 Total Daya Tampung per Program (Overall)")
fig2 = px.bar(
    df,
    x='Program Studi', y='Daya Tampung SNBT',
    title="Total Daya Tampung per Program",
    labels={'Program Studi': 'Program Studi', 'Daya Tampung SNBT': 'Jumlah Daya Tampung'},
    color='Daya Tampung SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig2)

# 3. **UKT Terendah vs UKT Tertinggi (Overall)**
st.subheader("📈 Perbandingan UKT Terendah dan Tertinggi per Program (Overall)")
fig3 = px.bar(
    df,
    x='Program Studi', y=['UKT Terendah', 'UKT Tertinggi'],
    title="Perbandingan UKT Terendah dan Tertinggi per Program",
    labels={'Program Studi': 'Program Studi', 'value': 'Biaya (IDR)'},
    color='variable', barmode='group'
)
st.plotly_chart(fig3)

# --- Initial Overview: Metrics for Each Province ---
st.header("📊 Data Overview per Province")

# 1. **Total Peminat per Province (Overall)**
st.subheader("📈 Total Peminat per Province (Overall)")
fig1 = px.bar(
    df,
    x='Provinsi', y='Peminat SNBT',
    title="Total Peminat per Province (Overall)",
    labels={'Provinsi': 'Province', 'Peminat SNBT': 'Jumlah Peminat'},
    color='Peminat SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig1)

# 2. **Total Daya Tampung per Province (Overall)**
st.subheader("📈 Total Daya Tampung per Province (Overall)")
fig2 = px.bar(
    df,
    x='Provinsi', y='Daya Tampung SNBT',
    title="Total Daya Tampung per Province (Overall)",
    labels={'Provinsi': 'Province', 'Daya Tampung SNBT': 'Jumlah Daya Tampung'},
    color='Daya Tampung SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig2)

# 3. **UKT Terendah vs UKT Tertinggi per Province (Overall)**
st.subheader("📈 Perbandingan UKT Terendah dan Tertinggi per Province (Overall)")
fig3 = px.bar(
    df,
    x='Provinsi', y=['UKT Terendah', 'UKT Tertinggi'],
    title="Perbandingan UKT Terendah dan Tertinggi per Province (Overall)",
    labels={'Provinsi': 'Province', 'value': 'Biaya (IDR)'},
    color='variable', barmode='group'
)
st.plotly_chart(fig3)


# --- Filter by Province ---
st.header("🎛 Filter by Province")
provinsi = st.selectbox("Pilih Provinsi:", df['Provinsi'].unique())

# Filter data based on the selected province
filtered_df = df[df['Provinsi'] == provinsi]

# --- Visualization 1: Total Peminat per Program in Selected Province ---
st.subheader(f"📊 Total Peminat per Program in {provinsi}")
fig4 = px.bar(
    filtered_df,
    x='Program Studi', y='Peminat SNBT',
    title=f"Total Peminat per Program in {provinsi}",
    labels={'Program Studi': 'Program Studi', 'Peminat SNBT': 'Jumlah Peminat'},
    color='Peminat SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig4)

# --- Visualization 2: Total Daya Tampung per Program in Selected Province ---
st.subheader(f"📊 Daya Tampung per Program in {provinsi}")
fig5 = px.bar(
    filtered_df,
    x='Program Studi', y='Daya Tampung SNBT',
    title=f"Daya Tampung per Program in {provinsi}",
    labels={'Program Studi': 'Program Studi', 'Daya Tampung SNBT': 'Jumlah Daya Tampung'},
    color='Daya Tampung SNBT', color_continuous_scale='Viridis'
)
st.plotly_chart(fig5)

# --- Visualization 3: UKT Terendah vs UKT Tertinggi per Program in Selected Province ---
st.subheader(f"📊 Perbandingan UKT Terendah dan Tertinggi per Program in {provinsi}")
fig6 = px.bar(
    filtered_df,
    x='Program Studi', y=['UKT Terendah', 'UKT Tertinggi'],
    title=f"Perbandingan UKT Terendah dan Tertinggi per Program in {provinsi}",
    labels={'Program Studi': 'Program Studi', 'value': 'Biaya (IDR)'},
    color='variable', barmode='group'
)
st.plotly_chart(fig6)



# --- Streamlit Title ---
st.title("📊 Dashboard Klasifikasi UKT")

# Load and preprocess the data
df = load_data()
df = preprocess_data(df)

# --- Form Input for Province, University, and UKT Category ---
st.header("🎯 Eksplorasi Prodi Berdasarkan Provinsi, Universitas, dan Kategori")

# 1. **Filter by Province**
provinsi = st.selectbox("Pilih Provinsi", df['Provinsi'].unique())

# Filter universities based on selected province
universitas_list = df[df['Provinsi'] == provinsi]['Universitas'].unique()
universitas = st.selectbox("Pilih Universitas", universitas_list)

# Filter programs based on selected university
program_studi_list = df[df['Universitas'] == universitas]['Program Studi'].unique()
program_studi = st.selectbox("Pilih Program Studi", program_studi_list)

# 2. **Filter by UKT Category**
kategori = st.selectbox("Pilih Kategori Jurusan", ['UKT Rendah', 'UKT Tinggi'])

# --- Filter data based on the selected criteria ---
if kategori == 'UKT Rendah':
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['UKT Terendah'] < np.percentile(df['UKT Terendah'], 25))]
else:
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['UKT Terendah'] >= np.percentile(df['UKT Terendah'], 25))]

# --- Display Data Based on Selected Filters ---
if not filtered_df.empty:
    st.success(f"✅ Menampilkan program studi di {universitas}, {provinsi} dengan kategori {kategori} (UKT Rendah):")
    st.write(f"Terdapat {len(filtered_df)} program studi yang memenuhi kriteria.")

    # Display the filtered data as a table
    st.dataframe(filtered_df[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah']])

    # Optionally, visualize some key metrics
    st.subheader("📊 Visualisasi Data Program")
    fig = px.bar(filtered_df, x='Program Studi', y=['Peminat SNBT', 'Daya Tampung SNBT', 'UKT Terendah'],
                 title=f"Analisis Program Studi di {universitas}, {provinsi}",
                 labels={'Program Studi': 'Program Studi', 'value': 'Jumlah'},
                 color='Program Studi', barmode='group')
    st.plotly_chart(fig)
else:
    st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")




# --- Streamlit Title ---
st.title("📊 Dashboard Klasifikasi Peminat")

# Load and preprocess the data
df = load_data()
df = preprocess_data(df)

# --- Eksplorasi Peminat Berdasarkan Provinsi, Universitas, dan Kategori ---
st.header("🎯 Eksplorasi Program Studi Berdasarkan Provinsi, Universitas, dan Kategori Peminat")

# 1. **Filter by Province**
provinsi = st.selectbox("Pilih Provinsi", df['Provinsi'].unique(), key='provinsi_selectbox')

# Filter universities based on selected province
universitas_list = df[df['Provinsi'] == provinsi]['Universitas'].unique()
universitas = st.selectbox("Pilih Universitas", universitas_list, key='universitas_selectbox')

# Filter programs based on selected university
program_studi_list = df[df['Universitas'] == universitas]['Program Studi'].unique()
program_studi = st.selectbox("Pilih Program Studi", program_studi_list, key='program_studi_selectbox')

# 2. **Filter by Peminat Category**
kategori_peminat = st.selectbox("Pilih Kategori Peminat", ['Peminat Banyak', 'Peminat Sedikit'], key='kategori_peminat_selectbox')

# --- Filter data based on the selected criteria ---
if kategori_peminat == 'Peminat Banyak':
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['Peminat SNBT'] >= np.percentile(df['Peminat SNBT'], 75))]  # Top 25% of applicants
else:
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['Peminat SNBT'] < np.percentile(df['Peminat SNBT'], 25))]  # Bottom 25% of applicants

# --- Display Data Based on Selected Filters ---
if not filtered_df.empty:
    st.success(f"✅ Menampilkan program studi di {universitas}, {provinsi} dengan kategori {kategori_peminat} (Peminat):")
    st.write(f"Terdapat {len(filtered_df)} program studi yang memenuhi kriteria.")

    # Display the filtered data as a table
    st.dataframe(filtered_df[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah']])

    # Optionally, visualize some key metrics
    st.subheader("📊 Visualisasi Data Program")
    fig = px.bar(filtered_df, x='Program Studi', y=['Peminat SNBT', 'Daya Tampung SNBT', 'UKT Terendah'],
                 title=f"Analisis Program Studi di {universitas}, {provinsi}",
                 labels={'Program Studi': 'Program Studi', 'value': 'Jumlah'},
                 color='Program Studi', barmode='group')
    st.plotly_chart(fig)
else:
    st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")







# --- Streamlit Title ---
st.title("📊 Dashboard Klasifikasi Daya Tampung")

# Load and preprocess the data
df = load_data()
df = preprocess_data(df)

# --- Eksplorasi Daya Tampung Berdasarkan Provinsi, Universitas, dan Kategori ---
st.header("🎯 Eksplorasi Program Studi Berdasarkan Provinsi, Universitas, dan Kategori Daya Tampung")

# 1. **Filter by Province**
provinsi = st.selectbox("Pilih Provinsi", df['Provinsi'].unique(), key=f'provinsi_selectbox_{hash("provinsi")}')
# Filter universities based on selected province
universitas_list = df[df['Provinsi'] == provinsi]['Universitas'].unique()
universitas = st.selectbox("Pilih Universitas", universitas_list, key=f'universitas_selectbox_{hash("universitas")}')
# Filter programs based on selected university
program_studi_list = df[df['Universitas'] == universitas]['Program Studi'].unique()
program_studi = st.selectbox("Pilih Program Studi", program_studi_list, key=f'program_studi_selectbox_{hash("program_studi")}')

# 2. **Filter by Daya Tampung Category**
kategori_daya_tampung = st.selectbox("Pilih Kategori Daya Tampung", ['Kuota Besar', 'Kuota Kecil'], key=f'kategori_daya_tampung_selectbox_{hash("kategori_daya_tampung")}')

# --- Filter data based on the selected criteria ---
if kategori_daya_tampung == 'Kuota Besar':
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['Daya Tampung SNBT'] >= np.percentile(df['Daya Tampung SNBT'], 75))]  # Top 25% of intake capacity
else:
    filtered_df = df[(df['Universitas'] == universitas) & 
                     (df['Provinsi'] == provinsi) & 
                     (df['Daya Tampung SNBT'] < np.percentile(df['Daya Tampung SNBT'], 25))]  # Bottom 25% of intake capacity

# --- Display Data Based on Selected Filters ---
if not filtered_df.empty:
    st.success(f"✅ Menampilkan program studi di {universitas}, {provinsi} dengan kategori {kategori_daya_tampung} (Daya Tampung):")
    st.write(f"Terdapat {len(filtered_df)} program studi yang memenuhi kriteria.")

    # Display the filtered data as a table
    st.dataframe(filtered_df[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah']])

else:
    st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")








# --- Streamlit Title ---
st.title("📊 Sistem Analisis Prodi")
import numpy as np


# Load and preprocess the data
df = load_data()
df = preprocess_data(df)

# --- Form Input for Universitas and Program Studi ---
st.header("🎯 Masukkan Data untuk Analisis Program Studi")
with st.form("form_input"):
    universitas = st.selectbox("Pilih Universitas", df['Universitas'].unique())

    # Filter program studi based on the selected universitas
    filtered_programs = df[df['Universitas'] == universitas]['Program Studi'].unique()
    program_studi = st.selectbox("Pilih Program Studi", filtered_programs)
    
    submit_button = st.form_submit_button("Analisis Program")

# --- Analisis Program Studi ---
if submit_button:
    # Filter the dataset based on the selected Universitas and Program Studi
    filtered_df = df[(df['Universitas'] == universitas) & (df['Program Studi'] == program_studi)]
    
    if not filtered_df.empty:
        # Get the data for the selected program
        program_data = filtered_df.iloc[0]
        
        peminat = program_data['Peminat SNBT']
        daya_tampung = program_data['Daya Tampung SNBT']
        ukt_terendah = program_data['UKT Terendah']
        ukt_tertinggi = program_data['UKT Tertinggi']
        
        # Calculate the 25th percentile for UKT Terendah and UKT Tertinggi
        ukt_terendah_25th = np.percentile(df['UKT Terendah'], 25)
        ukt_tertinggi_25th = np.percentile(df['UKT Tertinggi'], 25)
        
        # Calculate the range between UKT Terendah and UKT Tertinggi
        ukt_range = ukt_tertinggi - ukt_terendah
        
        # Create feedback based on the selected program
        feedback = ""
        
        # Check for low competition (fewer applicants)
        if peminat < np.percentile(df['Peminat SNBT'], 25):  # Below 25th percentile of peminat
            feedback += "📉 Program ini memiliki peminat yang sedikit, sehingga persaingan untuk diterima lebih rendah.\n"
        else:
            feedback += "📈 Program ini memiliki peminat yang banyak, persaingan lebih ketat.\n"
        
        # Check for high intake capacity (large quota)
        if daya_tampung > np.percentile(df['Daya Tampung SNBT'], 75):  # Above 75th percentile of daya tampung
            feedback += "📊 Program ini memiliki kuota daya tampung yang besar, peluang untuk diterima lebih tinggi.\n"
        else:
            feedback += "⚠️ Program ini memiliki kuota daya tampung yang terbatas.\n"
        
        # Check for low UKT (using both UKT Terendah and UKT Tertinggi)
        if ukt_terendah < ukt_terendah_25th and ukt_range < 5000000:  # Low UKT if within the 25th percentile and small range
            feedback += "💸 UKT program ini terjangkau, biaya kuliah rendah.\n"
        elif ukt_tertinggi < ukt_tertinggi_25th:  # If the highest fee is low
            feedback += "💸 UKT program ini terjangkau, biaya kuliah rendah.\n"
        else:
            feedback += "💰 UKT program ini relatif tinggi, biaya kuliah lebih mahal.\n"
        
        # Display the feedback
        st.success(f"✅ Hasil Analisis untuk Program {program_studi} di {universitas}:")
        st.write(feedback)
        
    else:
        st.error("Program studi yang Anda pilih tidak ditemukan.")
        



