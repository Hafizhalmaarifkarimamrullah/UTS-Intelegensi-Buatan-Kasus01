# UTS Intelegensi Buatan — Kasus 1: Klasifikasi Kultivar Anggur

- **Nama:** Hafizh Al Ma'arif Karim Amrullah
- **NIM:** 24552011137 (Jumlah digit = 31; 31 mod 10 = 1 → **Kasus 1**)
- **Kelas:** TIF RP 23D
- **Mata Kuliah:** Intelegensi Buatan (3 SKS) — Universitas Teknologi Bandung
- **Dosen:** Muhammad Ikhwan Fathulloh, S.Kom.
- **Dataset:** `kasus01_wine.csv` (Target: `kultivar` — `Kultivar A`, `Kultivar B`, `Kultivar C`)

## Struktur Proyek

```text
.
├── fitur.py                            # Audit data & transformer feature engineering (rasio fitur)
├── latih.py                            # Pelatihan, 5-Fold Stratified CV, evaluasi test 1x, simpan joblib
├── app.py                              # Aplikasi Streamlit (input mentah, probabilitas, metrik evaluasi)
├── requirements.txt                    # Daftar dependensi Python
├── dataset/
│   └── kasus01_wine.csv                # Diunduh otomatis oleh latih.py / Colab jika belum tersedia
├── models/
│   └── kasus01_wine_pipeline.joblib    # Dibuat otomatis saat latih.py dijalankan
└── notebook/
    └── kasus01_colab.ipynb             # Notebook Google Colab (training + jalankan Streamlit)
```

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
python latih.py
streamlit run app.py
```

## Cara Menjalankan di Google Colab

1. Buka `notebook/kasus01_colab.ipynb` di Google Colab.
2. Klik **Runtime → Run all**.
3. Notebook otomatis mengunduh `kasus01_wine.csv` dari repositori GitHub mata kuliah, melatih dan mengevaluasi model, menyimpan `models/kasus01_wine_pipeline.joblib`, serta menjalankan aplikasi Streamlit melalui `cloudflared` tunnel.
