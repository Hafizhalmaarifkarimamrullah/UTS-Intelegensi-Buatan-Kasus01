"""Modul Feature Engineering untuk Kasus 1: Klasifikasi Kultivar Anggur.

Nama  : Hafizh Al Ma'arif Karim Amrullah
NIM   : 24552011137 (Jumlah digit = 31, 31 mod 10 = 1 -> Kasus 1)
Kelas : TIF RP 23D
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FITUR_MENTAH: list[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

FITUR_RASIO: list[str] = [
    "rasio_flavanoid_fenol",
    "rasio_fenol_nonfenol",
    "rasio_warna_hue",
    "rasio_od_fenol",
    "rasio_prolin_alkohol",
]

KOLOM_TARGET: str = "kultivar"
EPS: float = 1e-6


def audit_data(df: pd.DataFrame, target_col: str = KOLOM_TARGET) -> dict:
    """Melakukan audit menyeluruh: dimensi, duplikat, missing value, tipe data, dan outlier IQR."""
    fitur_cols = [c for c in df.columns if c != target_col]
    outlier_per_kolom: dict[str, int] = {}

    for col in fitur_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            batas_bawah = q1 - 1.5 * iqr
            batas_atas = q3 + 1.5 * iqr
            outlier_per_kolom[col] = int(
                ((df[col] < batas_bawah) | (df[col] > batas_atas)).sum()
            )

    ringkasan = {
        "jumlah_baris": int(df.shape[0]),
        "jumlah_kolom": int(df.shape[1]),
        "jumlah_duplikat": int(df.duplicated().sum()),
        "missing_per_kolom": {col: int(val) for col, val in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "tipe_data": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "distribusi_target": (
            df[target_col].value_counts().to_dict() if target_col in df.columns else {}
        ),
        "outlier_iqr": outlier_per_kolom,
    }
    return ringkasan


def analisis_korelasi_tinggi(
    X_train: pd.DataFrame, threshold: float = 0.60
) -> list[dict]:
    """Mengidentifikasi pasangan fitur dengan |r| >= threshold pada data train saja."""
    corr = X_train[FITUR_MENTAH].corr()
    pasangan: list[dict] = []
    cols = FITUR_MENTAH

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_val = float(corr.iloc[i, j])
            if abs(r_val) >= threshold:
                pasangan.append(
                    {
                        "fitur_1": cols[i],
                        "fitur_2": cols[j],
                        "korelasi": round(r_val, 4),
                    }
                )

    pasangan.sort(key=lambda item: abs(item["korelasi"]), reverse=True)
    return pasangan


def buat_fitur_rasio(df: pd.DataFrame, drop_redundant: bool = False) -> pd.DataFrame:
    """Membuat 5 fitur rasio dari pasangan fitur kimia yang saling terkait.

    Pasangan fitur rasio:
    1. rasio_flavanoid_fenol : flavanoids / total_phenols
       Proporsi senyawa flavonoid terhadap total fenol.
    2. rasio_fenol_nonfenol  : flavanoids / nonflavanoid_phenols
       Keseimbangan senyawa fenolik flavonoid vs non-flavonoid.
    3. rasio_warna_hue       : color_intensity / hue
       Indeks kepekatan warna relatif terhadap rona warna anggur.
    4. rasio_od_fenol        : od280_od315_of_diluted_wines / total_phenols
       Rasio absorbansi UV terhadap total kadar fenol.
    5. rasio_prolin_alkohol  : proline / alcohol
       Keseimbangan asam amino prolin terhadap kadar alkohol.

    Parameter
    ---------
    drop_redundant : bool, default=False
        Jika True, membuang kolom 'total_phenols' yang berkorelasi sangat tinggi
        dengan 'flavanoids' dan sudah terwakili dalam fitur rasio.
    """
    hasil = df.copy()
    hasil = hasil[FITUR_MENTAH].astype(float)

    hasil["rasio_flavanoid_fenol"] = hasil["flavanoids"] / (hasil["total_phenols"] + EPS)
    hasil["rasio_fenol_nonfenol"] = hasil["flavanoids"] / (
        hasil["nonflavanoid_phenols"] + EPS
    )
    hasil["rasio_warna_hue"] = hasil["color_intensity"] / (hasil["hue"] + EPS)
    hasil["rasio_od_fenol"] = hasil["od280_od315_of_diluted_wines"] / (
        hasil["total_phenols"] + EPS
    )
    hasil["rasio_prolin_alkohol"] = hasil["proline"] / (hasil["alcohol"] + EPS)

    if drop_redundant:
        hasil = hasil.drop(columns=["total_phenols"])

    return hasil


class WineFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer scikit-learn agar feature engineering berjalan di dalam Pipeline.

    Keuntungan:
    - Menerima DataFrame mentah (13 kolom asli) secara langsung saat training maupun inferensi.
    - Bebas data leakage karena seluruh transformasi bersifat baris-per-baris sebelum scaling.
    """

    def __init__(self, drop_redundant: bool = False):
        self.drop_redundant = drop_redundant

    def fit(self, X: pd.DataFrame, y=None):
        self.fitur_masuk_ = list(FITUR_MENTAH)
        contoh = buat_fitur_rasio(
            pd.DataFrame(X, columns=FITUR_MENTAH).iloc[:1],
            drop_redundant=self.drop_redundant,
        )
        self.fitur_keluar_ = list(contoh.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=FITUR_MENTAH)
        else:
            X_df = X
        return buat_fitur_rasio(X_df, drop_redundant=self.drop_redundant)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.fitur_keluar_)
