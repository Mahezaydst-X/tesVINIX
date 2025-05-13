import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


# --- Load Dataset ---
@st.cache_data
def load_data():
    file_path = 'UKT_Kedokteran - Sheet1.csv'  # Pastikan file berada di folder yang sama
    data = pd.read_csv(file_path)
    return data

# --- Preprocess Data ---
def preprocess_data(df):
    # Membersihkan nama kolom
    df.columns = df.columns.str.strip()
    
    # Mengubah tipe data kolom yang relevan
    df['Universitas'] = df['Universitas'].astype(str)
    df['Program Studi'] = df['Program Studi'].astype(str)
    df['Provinsi'] = df['Provinsi'].astype(str)
    
    # Mengubah ke numerik, jika ada error, set menjadi NaN
    df['Daya Tampung SNBT'] = pd.to_numeric(df['Daya Tampung SNBT'], errors='coerce')
    df['Peminat SNBT'] = pd.to_numeric(df['Peminat SNBT'], errors='coerce')
    
    # Menangani kolom 'UKT Terendah' dan 'UKT Tertinggi'
    df['UKT Terendah'] = df['UKT Terendah'].str.replace('Rp', '').str.replace('.', '').astype(float)
    df['UKT Tertinggi'] = df['UKT Tertinggi'].str.replace('Rp', '').str.replace('.', '').astype(float)
    
    # Menghapus baris dengan nilai yang hilang (jika ada)
    df = df.dropna(subset=['Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah', 'UKT Tertinggi'])
    return df

# --- Judul Aplikasi Streamlit ---
st.set_page_config(page_title="Dashboard Kedokteran", page_icon="🏥", layout="wide")
st.title("🏥 Dashboard Klasifikasi Peminat dan Daya Tampung Fakultas Kedokteran")


# Informasi Tim dan Insight
st.markdown("""
<div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:25px; border-left: 5px solid #0066cc;">
    <h3 style="color:#0066cc;">📌 Insight: Perguruan Tinggi di Indonesia yang Memiliki Fakultas Kedokteran</h3>
    <p style="font-size:16px;">Program studi dengan kuota besar dan peminat sedikit cenderung menawarkan peluang diterima yang lebih tinggi. 
    Sementara itu, program studi dengan UKT rendah memberikan keuntungan bagi mahasiswa yang mencari biaya kuliah lebih terjangkau.</p>
    <hr style="border:1px dashed #ccc;">
    <p style="font-size:14px;"><strong>Disusun oleh Tim:</strong><br>
    I Gusti Ngurah Bagus Lanang Purbhawa (533) - Universitas Udayana<br>
    I Kadek Agus Candra Widnyana (534) - Universitas Udayana<br>
    I Kadek Angga Kusuma Diatmika (535) - Universitas Udayana<br>
    I Komang Maheza Yudistia (536) - Universitas Udayana
    </p>
</div>
""", unsafe_allow_html=True)

# Membuat Tab untuk memisahkan berbagai analisis
tab1, tab2, tab3, tab4 = st.tabs(["📊 Gambaran Umum", "🔍 Analisis UKT", "👨‍🎓 Analisis Peminat", "📋 Analisis Daya Tampung"])

# Load dan preprocess data
df = load_data()
df = preprocess_data(df)

# --- TAB 1: Gambaran Umum Data ---
with tab1:
    st.header("📊 Gambaran Umum Data")
    
    # Filter by Status Instansi
    status_options = ['Semua'] + sorted(df['Status Instansi (PTN/PTS)'].unique().tolist())
    status_filter = st.selectbox("Filter berdasarkan Status Instansi:", status_options)
    
    if status_filter != 'Semua':
        filtered_df = df[df['Status Instansi (PTN/PTS)'] == status_filter]
    else:
        filtered_df = df
    
    col1, col2 = st.columns(2)
    
    # Kolom 1: Visualisasi Total Peminat
    with col1:
        st.subheader("📈 Total Peminat per Program Studi")
        peminat_data = filtered_df.groupby('Program Studi')['Peminat SNBT'].sum().reset_index().sort_values('Peminat SNBT', ascending=False).head(10)
        fig1 = px.bar(
            peminat_data,
            x='Program Studi', y='Peminat SNBT',
            title="10 Program Studi dengan Peminat Terbanyak",
            labels={'Program Studi': 'Program Studi', 'Peminat SNBT': 'Jumlah Peminat'},
            color='Peminat SNBT', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Kolom 2: Visualisasi Total Daya Tampung
    with col2:
        st.subheader("📈 Total Daya Tampung per Program Studi")
        tampung_data = filtered_df.groupby('Program Studi')['Daya Tampung SNBT'].sum().reset_index().sort_values('Daya Tampung SNBT', ascending=False).head(10)
        fig2 = px.bar(
            tampung_data,
            x='Program Studi', y='Daya Tampung SNBT',
            title="10 Program Studi dengan Daya Tampung Terbanyak",
            labels={'Program Studi': 'Program Studi', 'Daya Tampung SNBT': 'Jumlah Daya Tampung'},
            color='Daya Tampung SNBT', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Perbandingan UKT
    st.subheader("📊 Perbandingan UKT Terendah dan Tertinggi per Program Studi")
    ukt_data = filtered_df.groupby('Program Studi')[['UKT Terendah', 'UKT Tertinggi']].mean().reset_index().sort_values('UKT Tertinggi', ascending=False).head(10)
    
    # Format currency untuk tampilan
    ukt_data['UKT Terendah (dalam Jutaan)'] = ukt_data['UKT Terendah'] / 1000000
    ukt_data['UKT Tertinggi (dalam Jutaan)'] = ukt_data['UKT Tertinggi'] / 1000000
    
    fig3 = px.bar(
        ukt_data,
        x='Program Studi', y=['UKT Terendah (dalam Jutaan)', 'UKT Tertinggi (dalam Jutaan)'],
        title="10 Program Studi dengan UKT Tertinggi",
        labels={'Program Studi': 'Program Studi', 'value': 'Biaya (Juta Rupiah)'},
        color_discrete_map={'UKT Terendah (dalam Jutaan)': '#2E86C1', 'UKT Tertinggi (dalam Jutaan)': '#E74C3C'},
        barmode='group'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Analisis per Provinsi
    st.header("📈 Analisis per Provinsi")
    
    # Filter by Province
    provinsi = st.selectbox("Pilih Provinsi untuk Analisis Detail:", df['Provinsi'].unique())
    
    # Filter data berdasarkan provinsi yang dipilih
    provinsi_df = df[df['Provinsi'] == provinsi]
    
    if not provinsi_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rata-rata UKT per Universitas dalam provinsi terpilih
            st.subheader(f"Rata-rata UKT per Universitas di {provinsi}")
            ukt_provinsi = provinsi_df.groupby('Universitas')[['UKT Terendah', 'UKT Tertinggi']].mean().reset_index()
            
            # Format currency untuk tampilan
            ukt_provinsi['UKT Terendah (dalam Jutaan)'] = ukt_provinsi['UKT Terendah'] / 1000000
            ukt_provinsi['UKT Tertinggi (dalam Jutaan)'] = ukt_provinsi['UKT Tertinggi'] / 1000000
            
            fig_ukt = px.bar(
                ukt_provinsi,
                x='Universitas', y=['UKT Terendah (dalam Jutaan)', 'UKT Tertinggi (dalam Jutaan)'],
                title=f"Rata-rata UKT per Universitas di {provinsi}",
                labels={'Universitas': 'Universitas', 'value': 'Biaya (Juta Rupiah)'},
                color_discrete_map={'UKT Terendah (dalam Jutaan)': '#2E86C1', 'UKT Tertinggi (dalam Jutaan)': '#E74C3C'},
                barmode='group'
            )
            st.plotly_chart(fig_ukt, use_container_width=True)
            
        with col2:
            # Total peminat per universitas dalam provinsi terpilih
            st.subheader(f"Total Peminat per Universitas di {provinsi}")
            peminat_provinsi = provinsi_df.groupby('Universitas')['Peminat SNBT'].sum().reset_index()
            fig_peminat = px.bar(
                peminat_provinsi,
                x='Universitas', y='Peminat SNBT',
                title=f"Total Peminat di {provinsi}",
                labels={'Universitas': 'Universitas', 'Peminat SNBT': 'Jumlah Peminat'},
                color='Peminat SNBT', color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_peminat, use_container_width=True)
            
    else:
        st.error(f"Tidak ada data untuk provinsi {provinsi}")


# --- TAB 2: Analisis UKT ---
with tab2:
    st.header("💰 Analisis UKT Program Studi")
    
    # Form Input untuk Provinsi, Universitas, dan Kategori UKT
    st.subheader("🔍 Eksplorasi Prodi Berdasarkan Provinsi, Universitas, dan Kategori UKT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Filter berdasarkan Provinsi
        provinsi_ukt = st.selectbox("Pilih Provinsi", df['Provinsi'].unique(), key="provinsi_ukt")
        
        # Filter universitas berdasarkan provinsi yang dipilih
        universitas_list = df[df['Provinsi'] == provinsi_ukt]['Universitas'].unique()
        universitas_ukt = st.selectbox("Pilih Universitas", universitas_list, key="universitas_ukt")
    
    with col2:
        # 2. Filter berdasarkan Kategori UKT
        kategori_ukt = st.selectbox("Pilih Kategori UKT", ['UKT Rendah', 'UKT Tinggi'], key="kategori_ukt")
        
        # Tombol untuk menjalankan analisis
        run_analysis_ukt = st.button("Analisis UKT", key="run_analysis_ukt")
    
    # Filter data berdasarkan kriteria yang dipilih
    if run_analysis_ukt:
        if kategori_ukt == 'UKT Rendah':
            # Ambil program studi dengan UKT di bawah persentil ke-25 (termurah 25%)
            threshold = np.percentile(df['UKT Terendah'], 25)
            filtered_ukt_df = df[(df['Universitas'] == universitas_ukt) & 
                             (df['Provinsi'] == provinsi_ukt) & 
                             (df['UKT Terendah'] < threshold)]
        else:
            # Ambil program studi dengan UKT di atas persentil ke-75 (termahal 25%)
            threshold = np.percentile(df['UKT Terendah'], 75)
            filtered_ukt_df = df[(df['Universitas'] == universitas_ukt) & 
                             (df['Provinsi'] == provinsi_ukt) & 
                             (df['UKT Terendah'] >= threshold)]
        
        # Tampilkan Data Berdasarkan Filter yang Dipilih
        if not filtered_ukt_df.empty:
            st.success(f"✅ Menampilkan program studi di {universitas_ukt}, {provinsi_ukt} dengan kategori {kategori_ukt}:")
            st.write(f"Terdapat {len(filtered_ukt_df)} program studi yang memenuhi kriteria.")
            
            # Format currency untuk tampilan yang lebih baik
            filtered_ukt_df_display = filtered_ukt_df.copy()
            filtered_ukt_df_display['UKT Terendah (Rp)'] = filtered_ukt_df_display['UKT Terendah'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
            filtered_ukt_df_display['UKT Tertinggi (Rp)'] = filtered_ukt_df_display['UKT Tertinggi'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
            
            # Tampilkan data sebagai tabel
            st.dataframe(filtered_ukt_df_display[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah (Rp)', 'UKT Tertinggi (Rp)']])
            
            # Visualisasi data program studi
            st.subheader("📊 Visualisasi Data Program Studi")
            
            # Siapkan data untuk visualisasi
            viz_data = filtered_ukt_df.copy()
            viz_data['UKT Terendah (dalam Jutaan)'] = viz_data['UKT Terendah'] / 1000000
            
            fig_ukt = px.bar(
                viz_data,
                x='Program Studi',
                y=['Peminat SNBT', 'Daya Tampung SNBT', 'UKT Terendah (dalam Jutaan)'],
                title=f"Analisis Program Studi di {universitas_ukt}, {provinsi_ukt}",
                labels={'Program Studi': 'Program Studi', 'value': 'Nilai', 'variable': 'Metrik'},
                barmode='group',
                color_discrete_map={
                    'Peminat SNBT': '#3498DB',
                    'Daya Tampung SNBT': '#2ECC71',
                    'UKT Terendah (dalam Jutaan)': '#E74C3C'
                }
            )
            st.plotly_chart(fig_ukt, use_container_width=True)
            
            # Tambahkan analisis perbandingan
            st.subheader("📈 Analisis Perbandingan")
            
            # Hitung rasio peminat/daya tampung
            filtered_ukt_df['Rasio Peminat/Daya Tampung'] = filtered_ukt_df['Peminat SNBT'] / filtered_ukt_df['Daya Tampung SNBT']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Visualisasi rasio peminat/daya tampung
                fig_rasio = px.bar(
                    filtered_ukt_df,
                    x='Program Studi', y='Rasio Peminat/Daya Tampung',
                    title="Rasio Peminat/Daya Tampung",
                    labels={'Program Studi': 'Program Studi', 'Rasio Peminat/Daya Tampung': 'Rasio'},
                    color='Rasio Peminat/Daya Tampung', color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_rasio, use_container_width=True)
            
            with col2:
                # Visualisasi UKT terendah dan tertinggi
                fig_ukt_range = px.bar(
                    filtered_ukt_df,
                    x='Program Studi', y=['UKT Terendah', 'UKT Tertinggi'],
                    title="Rentang UKT Program Studi",
                    labels={'Program Studi': 'Program Studi', 'value': 'UKT (Rp)', 'variable': 'Jenis UKT'},
                    barmode='group',
                    color_discrete_map={'UKT Terendah': '#2E86C1', 'UKT Tertinggi': '#E74C3C'}
                )
                st.plotly_chart(fig_ukt_range, use_container_width=True)
                
        else:
            st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")


# --- TAB 3: Analisis Peminat ---
with tab3:
    st.header("👨‍🎓 Analisis Peminat Program Studi")
    
    # Form Input untuk Provinsi, Universitas, dan Kategori Peminat
    st.subheader("🔍 Eksplorasi Program Studi Berdasarkan Peminat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Filter berdasarkan Provinsi
        provinsi_peminat = st.selectbox("Pilih Provinsi", df['Provinsi'].unique(), key="provinsi_peminat")
        
        # Filter universitas berdasarkan provinsi yang dipilih
        universitas_list_peminat = df[df['Provinsi'] == provinsi_peminat]['Universitas'].unique()
        universitas_peminat = st.selectbox("Pilih Universitas", universitas_list_peminat, key="universitas_peminat")
    
    with col2:
        # 2. Filter berdasarkan Kategori Peminat
        kategori_peminat = st.selectbox("Pilih Kategori Peminat", ['Peminat Banyak', 'Peminat Sedikit'], key="kategori_peminat")
        
        # Tombol untuk menjalankan analisis
        run_analysis_peminat = st.button("Analisis Peminat", key="run_analysis_peminat")
    
    # Filter data berdasarkan kriteria yang dipilih
    if run_analysis_peminat:
        if kategori_peminat == 'Peminat Banyak':
            # Ambil program studi dengan peminat di atas persentil ke-75 (peminat terbanyak 25%)
            threshold = np.percentile(df['Peminat SNBT'], 75)
            filtered_peminat_df = df[(df['Universitas'] == universitas_peminat) & 
                                 (df['Provinsi'] == provinsi_peminat) & 
                                 (df['Peminat SNBT'] >= threshold)]
        else:
            # Ambil program studi dengan peminat di bawah persentil ke-25 (peminat tersedikit 25%)
            threshold = np.percentile(df['Peminat SNBT'], 25)
            filtered_peminat_df = df[(df['Universitas'] == universitas_peminat) & 
                                 (df['Provinsi'] == provinsi_peminat) & 
                                 (df['Peminat SNBT'] < threshold)]
        
        # Tampilkan Data Berdasarkan Filter yang Dipilih
        if not filtered_peminat_df.empty:
            st.success(f"✅ Menampilkan program studi di {universitas_peminat}, {provinsi_peminat} dengan kategori {kategori_peminat}:")
            st.write(f"Terdapat {len(filtered_peminat_df)} program studi yang memenuhi kriteria.")
            
            # Format currency untuk tampilan yang lebih baik
            filtered_peminat_df_display = filtered_peminat_df.copy()
            filtered_peminat_df_display['UKT Terendah (Rp)'] = filtered_peminat_df_display['UKT Terendah'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
            
            # Tampilkan data sebagai tabel
            st.dataframe(filtered_peminat_df_display[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah (Rp)']])
            
            # Visualisasi perbandingan peminat vs daya tampung
            st.subheader("📊 Visualisasi Data Program Studi")
            fig_peminat = px.bar(
                filtered_peminat_df,
                x='Program Studi', y=['Peminat SNBT', 'Daya Tampung SNBT'],
                title=f"Perbandingan Peminat vs Daya Tampung di {universitas_peminat}, {provinsi_peminat}",
                labels={'Program Studi': 'Program Studi', 'value': 'Jumlah', 'variable': 'Kategori'},
                barmode='group',
                color_discrete_map={'Peminat SNBT': '#3498DB', 'Daya Tampung SNBT': '#2ECC71'}
            )
            st.plotly_chart(fig_peminat, use_container_width=True)
            
            # Visualisasi rasio peminat/daya tampung
            st.subheader("📈 Tingkat Persaingan Program Studi")
            filtered_peminat_df['Rasio Persaingan'] = filtered_peminat_df['Peminat SNBT'] / filtered_peminat_df['Daya Tampung SNBT']
            
            fig_rasio = px.bar(
                filtered_peminat_df.sort_values('Rasio Persaingan', ascending=False),
                x='Program Studi', y='Rasio Persaingan',
                title="Rasio Persaingan (Peminat/Daya Tampung)",
                labels={'Program Studi': 'Program Studi', 'Rasio Persaingan': 'Rasio Persaingan'},
                color='Rasio Persaingan', color_continuous_scale='RdYlGn_r'
            )
            fig_rasio.update_layout(yaxis_title="Rasio Persaingan (semakin tinggi semakin ketat)")
            st.plotly_chart(fig_rasio, use_container_width=True)
            
            # Tambahkan interpretasi
            st.info(f"""
            **Interpretasi Rasio Persaingan:**
            - Rasio > 10: Persaingan sangat ketat
            - Rasio 5-10: Persaingan ketat
            - Rasio 3-5: Persaingan sedang
            - Rasio < 3: Persaingan relatif rendah
            """)
            
        else:
            st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")


# --- TAB 4: Analisis Daya Tampung ---
with tab4:
    st.header("📋 Analisis Daya Tampung Program Studi")
    
    # Form Input untuk Provinsi, Universitas, dan Kategori Daya Tampung
    st.subheader("🔍 Eksplorasi Program Studi Berdasarkan Daya Tampung")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Filter berdasarkan Provinsi
        provinsi_daya = st.selectbox("Pilih Provinsi", df['Provinsi'].unique(), key="provinsi_daya")
        
        # Filter universitas berdasarkan provinsi yang dipilih
        universitas_list_daya = df[df['Provinsi'] == provinsi_daya]['Universitas'].unique()
        universitas_daya = st.selectbox("Pilih Universitas", universitas_list_daya, key="universitas_daya")
    
    with col2:
        # 2. Filter berdasarkan Kategori Daya Tampung
        kategori_daya = st.selectbox("Pilih Kategori Daya Tampung", ['Kuota Besar', 'Kuota Kecil'], key="kategori_daya")
        
        # Tombol untuk menjalankan analisis
        run_analysis_daya = st.button("Analisis Daya Tampung", key="run_analysis_daya")
    
    # Filter data berdasarkan kriteria yang dipilih
    if run_analysis_daya:
        if kategori_daya == 'Kuota Besar':
            # Ambil program studi dengan daya tampung di atas persentil ke-75 (kuota terbesar 25%)
            threshold = np.percentile(df['Daya Tampung SNBT'], 75)
            filtered_daya_df = df[(df['Universitas'] == universitas_daya) & 
                              (df['Provinsi'] == provinsi_daya) & 
                              (df['Daya Tampung SNBT'] >= threshold)]
        else:
            # Ambil program studi dengan daya tampung di bawah persentil ke-25 (kuota terkecil 25%)
            threshold = np.percentile(df['Daya Tampung SNBT'], 25)
            filtered_daya_df = df[(df['Universitas'] == universitas_daya) & 
                              (df['Provinsi'] == provinsi_daya) & 
                              (df['Daya Tampung SNBT'] < threshold)]
        
        # Tampilkan Data Berdasarkan Filter yang Dipilih
        if not filtered_daya_df.empty:
            st.success(f"✅ Menampilkan program studi di {universitas_daya}, {provinsi_daya} dengan kategori {kategori_daya}:")
            st.write(f"Terdapat {len(filtered_daya_df)} program studi yang memenuhi kriteria.")
            
            # Format currency untuk tampilan yang lebih baik
            filtered_daya_df_display = filtered_daya_df.copy()
            filtered_daya_df_display['UKT Terendah (Rp)'] = filtered_daya_df_display['UKT Terendah'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
            
            # Tampilkan data sebagai tabel
            st.dataframe(filtered_daya_df_display[['Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah (Rp)']])
            
            # Visualisasi data program studi
            st.subheader("📊 Kapasitas vs Peminat Program Studi")
            
            fig_daya = px.bar(
                filtered_daya_df,
                x='Program Studi', y=['Daya Tampung SNBT', 'Peminat SNBT'],
                title=f"Kapasitas vs Peminat di {universitas_daya}, {provinsi_daya}",
                labels={'Program Studi': 'Program Studi', 'value': 'Jumlah', 'variable': 'Kategori'},
                barmode='group',
                color_discrete_map={'Daya Tampung SNBT': '#2ECC71', 'Peminat SNBT': '#3498DB'}
            )
            st.plotly_chart(fig_daya, use_container_width=True)
            
            # Hitung persentase pemenuhan kuota
            filtered_daya_df['Persentase Pemenuhan'] = (filtered_daya_df['Peminat SNBT'] / filtered_daya_df['Daya Tampung SNBT']) * 100
                        
            st.subheader("📈 Persentase Pemenuhan Kuota")
            fig_pemenuhan = px.bar(
                filtered_daya_df,
                x='Program Studi', y='Persentase Pemenuhan',
                title="Persentase Pemenuhan Kuota",
                labels={'Program Studi': 'Program Studi', 'Persentase Pemenuhan': 'Persentase (%)'},
                color='Persentase Pemenuhan', color_continuous_scale='RdYlGn'
            )
            fig_pemenuhan.update_layout(yaxis_title="Pemenuhan Kuota (%)")
            st.plotly_chart(fig_pemenuhan, use_container_width=True)

# Tambahkan model prediksi jika ada cukup data
            if len(filtered_daya_df) >= 3:
                st.subheader("🔮 Model Prediksi Peluang Diterima")
                
                # Buat fitur sederhana
                X = filtered_daya_df[['Daya Tampung SNBT', 'Peminat SNBT', 'UKT Terendah']].values
                
                # Buat label kategori (0: Sulit diterima, 1: Peluang sedang, 2: Mudah diterima)
                filtered_daya_df['Rasio'] = filtered_daya_df['Peminat SNBT'] / filtered_daya_df['Daya Tampung SNBT']
                
                def get_label(rasio):
                    if rasio > 10:
                        return 0  # Sulit diterima
                    elif rasio > 5:
                        return 1  # Peluang sedang
                    else:
                        return 2  # Mudah diterima
                
                filtered_daya_df['Label'] = filtered_daya_df['Rasio'].apply(get_label)
                y = filtered_daya_df['Label'].values
                
                # Buat model
                try:
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
                    model.fit(X, y)
                    
                    # Tampilkan hasil prediksi peluang diterima untuk setiap program studi
                    st.write("Hasil Prediksi Peluang Diterima:")
                    
                    # Map hasil prediksi ke label yang lebih informatif
                    label_map = {0: "Sulit", 1: "Sedang", 2: "Mudah"}
                    
                    # Buat dataframe untuk visualisasi hasil prediksi
                    pred_df = filtered_daya_df[['Program Studi', 'Rasio']].copy()
                    pred_df['Prediksi Peluang'] = model.predict(X)
                    pred_df['Prediksi Peluang'] = pred_df['Prediksi Peluang'].map(label_map)
                    
                    # Visualisasi hasil prediksi
                    fig_pred = px.bar(
                        pred_df,
                        x='Program Studi', y='Rasio',
                        title="Prediksi Peluang Diterima per Program Studi",
                        labels={'Program Studi': 'Program Studi', 'Rasio': 'Rasio Persaingan'},
                        color='Prediksi Peluang',
                        color_discrete_map={'Sulit': '#E74C3C', 'Sedang': '#F39C12', 'Mudah': '#2ECC71'}
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Tampilkan tabel hasil prediksi
                    st.dataframe(pred_df)
                    
                    # Keterangan tambahan tentang prediksi
                    st.info("""
                    **Keterangan Prediksi:**
                    - 🔴 Sulit: Persaingan sangat ketat dengan rasio peminat/daya tampung > 10
                    - 🟠 Sedang: Persaingan moderat dengan rasio peminat/daya tampung 5-10
                    - 🟢 Mudah: Persaingan relatif rendah dengan rasio peminat/daya tampung < 5
                    """)
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam pembuatan model prediksi: {e}")
                
        else:
            st.error("Tidak ada program studi yang memenuhi kriteria yang Anda pilih.")

# --- Fitur Tambahan: Rekomendasi Program Studi ---
st.header("🎯 Rekomendasi Program Studi")
st.subheader("Cari program studi yang sesuai dengan preferensi Anda")

# Form untuk input preferensi pengguna
with st.form("form_rekomendasi"):
    st.write("Masukkan preferensi Anda untuk mendapatkan rekomendasi program studi:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pilih Provinsi
        prov_rekomendasi = st.selectbox("Provinsi", df['Provinsi'].unique(), key="prov_rekomendasi")
        
        # Rentang UKT
        ukt_min, ukt_max = st.slider(
            "Rentang UKT (dalam juta Rupiah)",
            float(df['UKT Terendah'].min() / 1000000),
            float(df['UKT Tertinggi'].max() / 1000000),
            (float(df['UKT Terendah'].min() / 1000000), float(df['UKT Tertinggi'].max() / 1000000)),
            format="Rp%.1f juta"
        )
    
    with col2:
        # Preferensi tingkat persaingan
        persaingan = st.radio(
            "Preferensi Tingkat Persaingan",
            ["Rendah", "Sedang", "Tinggi"],
            key="persaingan"
        )
        
        # Preferensi ukuran daya tampung
        ukuran_tampung = st.radio(
            "Preferensi Ukuran Daya Tampung",
            ["Kecil", "Sedang", "Besar"],
            key="ukuran_tampung"
        )
    
    # Tombol submit
    submit_button = st.form_submit_button("Cari Rekomendasi")

# Proses rekomendasi ketika tombol submit ditekan
if submit_button:
    # Filter berdasarkan provinsi
    rekomendasi_df = df[df['Provinsi'] == prov_rekomendasi].copy()
    
    # Filter berdasarkan rentang UKT (dalam juta)
    ukt_min_value = ukt_min * 1000000
    ukt_max_value = ukt_max * 1000000
    rekomendasi_df = rekomendasi_df[(rekomendasi_df['UKT Terendah'] >= ukt_min_value) & 
                                    (rekomendasi_df['UKT Terendah'] <= ukt_max_value)]
    
    # Hitung rasio persaingan
    rekomendasi_df['Rasio Persaingan'] = rekomendasi_df['Peminat SNBT'] / rekomendasi_df['Daya Tampung SNBT']
    
    # Filter berdasarkan tingkat persaingan
    if persaingan == "Rendah":
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Rasio Persaingan'] < 5]
    elif persaingan == "Sedang":
        rekomendasi_df = rekomendasi_df[(rekomendasi_df['Rasio Persaingan'] >= 5) & (rekomendasi_df['Rasio Persaingan'] < 10)]
    else:  # Tinggi
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Rasio Persaingan'] >= 10]
    
    # Filter berdasarkan ukuran daya tampung
    q25 = np.percentile(df['Daya Tampung SNBT'], 25)
    q75 = np.percentile(df['Daya Tampung SNBT'], 75)
    
    if ukuran_tampung == "Kecil":
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Daya Tampung SNBT'] < q25]
    elif ukuran_tampung == "Sedang":
        rekomendasi_df = rekomendasi_df[(rekomendasi_df['Daya Tampung SNBT'] >= q25) & (rekomendasi_df['Daya Tampung SNBT'] < q75)]
    else:  # Besar
        rekomendasi_df = rekomendasi_df[rekomendasi_df['Daya Tampung SNBT'] >= q75]
    
    # Tampilkan hasil rekomendasi
    if not rekomendasi_df.empty:
        st.success(f"✅ Ditemukan {len(rekomendasi_df)} program studi yang sesuai dengan preferensi Anda!")
        
        # Format tampilan
        rekomendasi_df_display = rekomendasi_df.copy()
        rekomendasi_df_display['UKT Terendah (Rp)'] = rekomendasi_df_display['UKT Terendah'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        rekomendasi_df_display['UKT Tertinggi (Rp)'] = rekomendasi_df_display['UKT Tertinggi'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        rekomendasi_df_display['Rasio Persaingan'] = rekomendasi_df_display['Rasio Persaingan'].round(2)
        
        # Tampilkan tabel hasil rekomendasi
        st.dataframe(rekomendasi_df_display[['Universitas', 'Program Studi', 'Daya Tampung SNBT', 'Peminat SNBT', 'Rasio Persaingan', 'UKT Terendah (Rp)', 'UKT Tertinggi (Rp)']])
        
        # Visualisasi hasil rekomendasi
        st.subheader("📊 Visualisasi Program Studi yang Direkomendasikan")
        
        # Visualisasi perbandingan rasio persaingan
        fig_rekomendasi = px.bar(
            rekomendasi_df.sort_values('Rasio Persaingan', ascending=True),
            x='Program Studi', y='Rasio Persaingan',
            title="Perbandingan Rasio Persaingan Program Studi yang Direkomendasikan",
            labels={'Program Studi': 'Program Studi', 'Rasio Persaingan': 'Rasio Persaingan'},
            color='Rasio Persaingan', color_continuous_scale='RdYlGn_r',
            hover_data=['Universitas', 'Daya Tampung SNBT', 'Peminat SNBT']
        )
        st.plotly_chart(fig_rekomendasi, use_container_width=True)
        
        # Visualisasi perbandingan UKT
        rekomendasi_df['UKT Terendah (dalam Jutaan)'] = rekomendasi_df['UKT Terendah'] / 1000000
        rekomendasi_df['UKT Tertinggi (dalam Jutaan)'] = rekomendasi_df['UKT Tertinggi'] / 1000000
        
        fig_ukt_rekomendasi = px.bar(
            rekomendasi_df.sort_values('UKT Terendah'),
            x='Program Studi', y=['UKT Terendah (dalam Jutaan)', 'UKT Tertinggi (dalam Jutaan)'],
            title="Perbandingan UKT Program Studi yang Direkomendasikan",
            labels={'Program Studi': 'Program Studi', 'value': 'UKT (dalam Jutaan)', 'variable': 'Jenis UKT'},
            barmode='group',
            color_discrete_map={'UKT Terendah (dalam Jutaan)': '#2E86C1', 'UKT Tertinggi (dalam Jutaan)': '#E74C3C'},
            hover_data=['Universitas']
        )
        st.plotly_chart(fig_ukt_rekomendasi, use_container_width=True)
        
        # Top 3 rekomendasi
        st.subheader("🏆 Top 3 Rekomendasi untuk Anda")
        
        # Buat skor berdasarkan preferensi
        # Normalisasi fitur
        rekomendasi_df['UKT_norm'] = 1 - ((rekomendasi_df['UKT Terendah'] - rekomendasi_df['UKT Terendah'].min()) / 
                                        (rekomendasi_df['UKT Terendah'].max() - rekomendasi_df['UKT Terendah'].min()))
        
        if persaingan == "Rendah":
            rekomendasi_df['Persaingan_score'] = 1 - (rekomendasi_df['Rasio Persaingan'] / rekomendasi_df['Rasio Persaingan'].max())
        elif persaingan == "Tinggi":
            rekomendasi_df['Persaingan_score'] = rekomendasi_df['Rasio Persaingan'] / rekomendasi_df['Rasio Persaingan'].max()
        else:  # Sedang
            rekomendasi_df['Persaingan_score'] = 1 - abs((rekomendasi_df['Rasio Persaingan'] - 7.5) / 7.5)
        
        # Skor total
        rekomendasi_df['Total_Score'] = (0.4 * rekomendasi_df['UKT_norm'] + 
                                       0.6 * rekomendasi_df['Persaingan_score'])
        
        # Top 3 berdasarkan skor
        top3 = rekomendasi_df.sort_values('Total_Score', ascending=False).head(3)
        
        # Tampilkan top 3 dalam format card
        col1, col2, col3 = st.columns(3)
        
        for i, (_, row) in enumerate(top3.iterrows()):
            col = [col1, col2, col3][i]
            with col:
                st.markdown(f"""
                <div style="border:1px solid #ccc; padding:10px; border-radius:10px; text-align:center; height:100%;">
                    <h3 style="color:#0066cc;">#{i+1}: {row['Program Studi']}</h3>
                    <p><strong>Universitas:</strong> {row['Universitas']}</p>
                    <p><strong>UKT:</strong> Rp{row['UKT Terendah']:,.0f}</p>
                    <p><strong>Daya Tampung:</strong> {int(row['Daya Tampung SNBT'])}</p>
                    <p><strong>Rasio Persaingan:</strong> {row['Rasio Persaingan']:.2f}</p>
                    <p><strong>Skor Kesesuaian:</strong> {row['Total_Score']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.error("Tidak ditemukan program studi yang sesuai dengan preferensi Anda. Coba sesuaikan preferensi Anda.")

# Tambahkan footer
st.markdown("""
<div style="text-align:center; margin-top:30px; padding:20px; background-color:#f0f2f6; border-radius:10px;">
    <h3>Tentang Dashboard</h3>
    <p>Dashboard ini dikembangkan untuk membantu calon mahasiswa memilih program studi kedokteran yang sesuai dengan preferensi mereka.</p>
    <p>Data yang digunakan dalam dashboard ini diperoleh dari sumber terpercaya tentang Universitas Kedokteran di Indonesia.</p>
    <p>© 2024 Universitas Udayana - Kelompok UKT Kedokteran</p>
</div>
""", unsafe_allow_html=True)
