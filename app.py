"""Aplikasi Streamlit Klasifikasi Kultivar Anggur - UTS Intelegensi Buatan (Kasus 1).

Nama  : Hafizh Al Ma'arif Karim Amrullah
NIM   : 24552011137 (Kasus 1: kasus01_wine.csv)
Kelas : TIF RP 23D
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Import modul fitur agar custom transformer WineFeatureEngineer dikenali saat joblib.load()
import fitur  # noqa: F401
from fitur import FITUR_MENTAH, buat_fitur_rasio

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "kasus01_wine_pipeline.joblib"

LABEL_FITUR: dict[str, str] = {
    "alcohol": "Kadar Alkohol (%)",
    "malic_acid": "Asam Malat (g/L)",
    "ash": "Kadar Abu (g/L)",
    "alcalinity_of_ash": "Alkalinitas Abu",
    "magnesium": "Magnesium (mg/L)",
    "total_phenols": "Total Fenol",
    "flavanoids": "Flavonoid",
    "nonflavanoid_phenols": "Fenol Non-Flavonoid",
    "proanthocyanins": "Proantosianin",
    "color_intensity": "Intensitas Warna",
    "hue": "Rona Warna (Hue)",
    "od280_od315_of_diluted_wines": "Rasio Absorbansi OD280/OD315",
    "proline": "Kadar Prolin (mg/L)",
}

st.set_page_config(
    page_title="Klasifikasi Kultivar Anggur — UTS AI (Kasus 1)",
    page_icon="🍷",
    layout="wide",
)


@st.cache_resource
def muat_bundle() -> dict:
    if not MODEL_PATH.exists():
        st.error(
            "File model `models/kasus01_wine_pipeline.joblib` belum ditemukan. "
            "Jalankan `python latih.py` terlebih dahulu."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


bundle = muat_bundle()
pipeline = bundle["pipeline"]
kelas_list: list[str] = bundle["kelas"]
stat_input: dict = bundle["statistik_input"]
contoh_sampel: dict = bundle["contoh_sampel"]
metrik_test: dict = bundle["metrik_test"]
metrik_cv: dict = bundle["metrik_cv"]

st.title("Klasifikasi Kultivar Anggur dari Analisis Kimia")
st.caption(
    f"UTS Intelegensi Buatan — Kasus 1 (`kasus01_wine.csv`) | "
    f"Hafizh Al Ma'arif Karim Amrullah (NIM 24552011137, TIF RP 23D) | "
    f"Model aktif: **{bundle['nama_model_terbaik']}**"
)

tab_prediksi, tab_evaluasi, tab_fe = st.tabs(
    [
        "1. Prediksi Sampel (Input Mentah)",
        "2. Metrik Evaluasi & Perbandingan Model",
        "3. Feature Engineering & Metadata",
    ]
)

with tab_prediksi:
    st.subheader("Masukkan 13 Komponen Kimia Sampel Anggur")
    st.write(
        "Aplikasi menerima **input mentah** secara langsung. Transformasi fitur rasio, "
        "imputasi median, dan `StandardScaler` dijalankan otomatis di dalam pipeline."
    )

    # Preset contoh cepat untuk pengujian
    if "nilai_input" not in st.session_state:
        st.session_state["nilai_input"] = {
            col: float(stat_input[col]["median"]) for col in FITUR_MENTAH
        }

    preset_cols = st.columns(len(kelas_list) + 1)
    for idx, nama_kelas in enumerate(kelas_list):
        if preset_cols[idx].button(f"Contoh {nama_kelas}", use_container_width=True):
            st.session_state["nilai_input"] = {
                col: float(contoh_sampel[nama_kelas][col]) for col in FITUR_MENTAH
            }
    if preset_cols[-1].button("Reset ke Median", use_container_width=True):
        st.session_state["nilai_input"] = {
            col: float(stat_input[col]["median"]) for col in FITUR_MENTAH
        }

    col1, col2, col3 = st.columns(3)
    grup_1 = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium"]
    grup_2 = ["total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins"]
    grup_3 = ["color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"]

    input_pengguna: dict[str, float] = {}

    with col1:
        st.markdown("**Komposisi Dasar & Mineral**")
        for col in grup_1:
            input_pengguna[col] = st.number_input(
                LABEL_FITUR[col],
                min_value=0.0,
                max_value=float(stat_input[col]["max"] * 2.0),
                value=float(st.session_state["nilai_input"][col]),
                step=0.1 if col != "magnesium" else 1.0,
                format="%.3f",
                key=f"in_{col}",
            )

    with col2:
        st.markdown("**Komponen Fenolik**")
        for col in grup_2:
            input_pengguna[col] = st.number_input(
                LABEL_FITUR[col],
                min_value=0.0,
                max_value=float(stat_input[col]["max"] * 2.0),
                value=float(st.session_state["nilai_input"][col]),
                step=0.05,
                format="%.3f",
                key=f"in_{col}",
            )

    with col3:
        st.markdown("**Warna, Absorbansi & Prolin**")
        for col in grup_3:
            input_pengguna[col] = st.number_input(
                LABEL_FITUR[col],
                min_value=0.0,
                max_value=float(stat_input[col]["max"] * 2.0),
                value=float(st.session_state["nilai_input"][col]),
                step=10.0 if col == "proline" else 0.05,
                format="%.3f",
                key=f"in_{col}",
            )

    if st.button("Prediksi Kultivar", type="primary", use_container_width=True):
        df_mentah = pd.DataFrame([input_pengguna])[bundle["fitur"]]
        prediksi = pipeline.predict(df_mentah)[0]
        probabilitas = pipeline.predict_proba(df_mentah)[0]
        prob_tertinggi = float(probabilitas.max())

        st.divider()
        res_kiri, res_kanan = st.columns([1, 1.3])

        with res_kiri:
            st.success(f"Hasil Prediksi: **{prediksi}**")
            st.metric("Probabilitas Kelas Terpilih", f"{prob_tertinggi * 100:.2f}%")
            df_prob = pd.DataFrame(
                {
                    "Kultivar": kelas_list,
                    "Probabilitas": [round(float(p), 4) for p in probabilitas],
                    "Persentase (%)": [f"{p * 100:.2f}%" for p in probabilitas],
                }
            )
            st.dataframe(df_prob, hide_index=True, use_container_width=True)

        with res_kanan:
            st.markdown("**Distribusi Probabilitas per Kultivar**")
            chart_df = pd.DataFrame(
                {"Probabilitas": probabilitas}, index=kelas_list
            )
            st.bar_chart(chart_df)

        with st.expander("Lihat 5 Fitur Rasio Turunan yang Dihitung Otomatis"):
            df_rasio = buat_fitur_rasio(df_mentah)[bundle["fitur_rasio"]]
            st.dataframe(df_rasio.round(4), hide_index=True, use_container_width=True)

with tab_evaluasi:
    st.subheader("Hasil Evaluasi Final pada Data Test (Dieksekusi 1 Kali)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{metrik_test['accuracy'] * 100:.2f}%")
    m2.metric("Test Macro F1", f"{metrik_test['f1_macro'] * 100:.2f}%")
    m3.metric("Test Weighted F1", f"{metrik_test['f1_weighted'] * 100:.2f}%")
    m4.metric("ROC-AUC (OvR Macro)", f"{metrik_test['roc_auc_ovr_macro'] * 100:.2f}%")

    st.subheader("Perbandingan Model (5-Fold Stratified CV pada Data Train)")
    df_perbandingan = pd.DataFrame(bundle["perbandingan_model"])
    st.dataframe(df_perbandingan, hide_index=True, use_container_width=True)

    c_kiri, c_kanan = st.columns(2)
    with c_kiri:
        st.subheader("Confusion Matrix (Data Test)")
        df_cm = pd.DataFrame(
            bundle["confusion_matrix"],
            index=[f"Aktual: {k}" for k in kelas_list],
            columns=[f"Pred: {k}" for k in kelas_list],
        )
        st.dataframe(df_cm, use_container_width=True)

    with c_kanan:
        st.subheader("Metrik per Kelas (Data Test)")
        rows_cls = []
        for k in kelas_list:
            info = bundle["classification_report"][k]
            rows_cls.append(
                {
                    "Kelas": k,
                    "Precision": round(info["precision"], 4),
                    "Recall": round(info["recall"], 4),
                    "F1-Score": round(info["f1-score"], 4),
                    "Support": int(info["support"]),
                }
            )
        st.dataframe(pd.DataFrame(rows_cls), hide_index=True, use_container_width=True)

with tab_fe:
    st.subheader("Pasangan Fitur Berkorelasi Tinggi (Data Train, |r| >= 0.60)")
    st.dataframe(
        pd.DataFrame(bundle["korelasi_tinggi"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Metadata Pipeline Tersimpan")
    st.json(
        {
            "nama_model_terbaik": bundle["nama_model_terbaik"],
            "jumlah_fitur_mentah": len(bundle["fitur"]),
            "fitur_rasio_tambahan": bundle["fitur_rasio"],
            "kelas_target": bundle["kelas"],
            "cv_f1_macro_mean": metrik_cv["cv_f1_macro_mean"],
            "versi_sklearn": bundle["versi_sklearn"],
        }
    )
