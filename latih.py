"""Skrip Pelatihan, Validasi Silang, Evaluasi, dan Penyimpanan Model - Kasus 1 (Wine).

Nama  : Hafizh Al Ma'arif Karim Amrullah
NIM   : 24552011137 (Jumlah digit = 31, 31 mod 10 = 1 -> Kasus 1)
Kelas : TIF RP 23D

Aturan yang dipatuhi:
1. Split train/test dilakukan SEBELUM imputasi, scaling, atau seleksi fitur.
2. Pemilihan model & keputusan drop/keep fitur dilakukan dengan 5-Fold Stratified CV
   pada data train saja.
3. Evaluasi pada data test dilakukan tepat SATU KALI untuk model terbaik.
4. Pipeline lengkap + metadata disimpan ke models/kasus01_wine_pipeline.joblib tanpa path absolut.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fitur import (
    FITUR_MENTAH,
    FITUR_RASIO,
    KOLOM_TARGET,
    WineFeatureEngineer,
    analisis_korelasi_tinggi,
    audit_data,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "kasus01_wine.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "kasus01_wine_pipeline.joblib"
SUMMARY_PATH = MODELS_DIR / "ringkasan_evaluasi.json"

GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/Muhammad-Ikhwan-Fathulloh/"
    "Artificial-Intelegent-Course-Bank/main/UTS/dataset/kasus01_wine.csv"
)


def muat_dataset() -> pd.DataFrame:
    """Memuat dataset lokal dari dataset/kasus01_wine.csv (atau mengunduh jika belum ada)."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(GITHUB_RAW_URL)
    df.to_csv(DATA_PATH, index=False)
    return df


def buat_kandidat_pipeline() -> dict[str, Pipeline]:
    """Menyusun kandidat pipeline klasifikasi dengan imputasi dan StandardScaler."""
    return {
        "Logistic Regression (Keep + Rasio)": Pipeline(
            [
                ("fe", WineFeatureEngineer(drop_redundant=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]
        ),
        "Logistic Regression (Drop Redundan + Rasio)": Pipeline(
            [
                ("fe", WineFeatureEngineer(drop_redundant=True)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]
        ),
        "Random Forest (Keep + Rasio)": Pipeline(
            [
                ("fe", WineFeatureEngineer(drop_redundant=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM RBF (Keep + Rasio)": Pipeline(
            [
                ("fe", WineFeatureEngineer(drop_redundant=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
            ]
        ),
    }


def hitung_statistik_input(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, dict]:
    """Menghitung statistik input & contoh sampel per kultivar dari X_train saja."""
    statistik: dict[str, dict[str, float]] = {}
    for col in FITUR_MENTAH:
        statistik[col] = {
            "min": round(float(X_train[col].min()), 3),
            "median": round(float(X_train[col].median()), 3),
            "mean": round(float(X_train[col].mean()), 3),
            "max": round(float(X_train[col].max()), 3),
            "std": round(float(X_train[col].std()), 3),
        }

    contoh_sampel: dict[str, dict[str, float]] = {}
    for kelas in sorted(y_train.unique().tolist()):
        baris_kelas = X_train.loc[y_train == kelas].median()
        contoh_sampel[kelas] = {
            col: round(float(baris_kelas[col]), 3) for col in FITUR_MENTAH
        }

    return statistik, contoh_sampel


def main() -> None:
    print("=" * 72)
    print("PELATIHAN MODEL KLASIFIKASI - KASUS 1: KULTIVAR ANGGUR (kasus01_wine.csv)")
    print("=" * 72)

    # 1. Muat & audit dataset
    df = muat_dataset()
    hasil_audit = audit_data(df, target_col=KOLOM_TARGET)
    print(f"\n[1] Audit Dataset:")
    print(f"    - Dimensi         : {hasil_audit['jumlah_baris']} baris x {hasil_audit['jumlah_kolom']} kolom")
    print(f"    - Duplikat        : {hasil_audit['jumlah_duplikat']}")
    print(f"    - Total Missing   : {hasil_audit['total_missing']}")
    print(f"    - Distribusi Kelas: {hasil_audit['distribusi_target']}")

    # 2. Split train dan test SEBELUM preprocessing (anti data leakage)
    X = df[FITUR_MENTAH].copy()
    y = df[KOLOM_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Pembagian Data (Stratified 80/20 sebelum preprocessing):")
    print(f"    - Train: {len(X_train)} baris | Test: {len(X_test)} baris")

    # 3. Analisis korelasi antarfitur pada data train saja
    korelasi_tinggi = analisis_korelasi_tinggi(X_train, threshold=0.60)
    print("\n[3] Pasangan Fitur Berkorelasi Tinggi pada Data Train (|r| >= 0.60):")
    for item in korelasi_tinggi:
        print(f"    - {item['fitur_1']:30s} <-> {item['fitur_2']:30s}: r = {item['korelasi']:+.4f}")

    # 4. Perbandingan model menggunakan 5-Fold Stratified Cross-Validation di data train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kandidat = buat_kandidat_pipeline()
    perbandingan_model: list[dict] = []

    print("\n[4] Perbandingan Model (5-Fold Stratified CV pada Data Train):")
    for nama, pipe in kandidat.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            n_jobs=-1,
        )
        ringkas = {
            "model": nama,
            "cv_accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
            "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
            "cv_f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
            "cv_f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
            "cv_f1_weighted_mean": round(float(scores["test_f1_weighted"].mean()), 4),
        }
        perbandingan_model.append(ringkas)
        print(
            f"    - {nama:44s} | Acc: {ringkas['cv_accuracy_mean']:.4f} (+/- {ringkas['cv_accuracy_std']:.4f}) "
            f"| F1-Macro: {ringkas['cv_f1_macro_mean']:.4f} (+/- {ringkas['cv_f1_macro_std']:.4f})"
        )

    # Pilih model terbaik berdasarkan CV F1-Macro, lalu CV Accuracy, lalu std terkecil
    perbandingan_model.sort(
        key=lambda r: (r["cv_f1_macro_mean"], r["cv_accuracy_mean"], -r["cv_f1_macro_std"]),
        reverse=True,
    )
    terbaik_info = perbandingan_model[0]
    nama_terbaik = terbaik_info["model"]
    pipeline_terbaik = kandidat[nama_terbaik]
    print(f"\n    ==> Model Terbaik (berdasarkan CV Data Train): {nama_terbaik}")

    # 5. Latih ulang model terbaik pada seluruh data train, evaluasi 1 KALI di data test
    pipeline_terbaik.fit(X_train, y_train)
    y_pred = pipeline_terbaik.predict(X_test)
    y_proba = pipeline_terbaik.predict_proba(X_test)
    kelas = list(pipeline_terbaik.classes_)

    metrik_test = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "roc_auc_ovr_macro": round(
            float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")), 4
        ),
    }
    cm = confusion_matrix(y_test, y_pred, labels=kelas)
    cls_report_dict = classification_report(
        y_test, y_pred, target_names=kelas, output_dict=True
    )

    print("\n[5] Evaluasi Final pada Data Test (Dieksekusi Tepat 1 Kali):")
    for k, v in metrik_test.items():
        print(f"    - {k:20s}: {v:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=kelas))
    print("    Confusion Matrix (baris=Aktual, kolom=Prediksi):")
    print(pd.DataFrame(cm, index=kelas, columns=kelas))

    # 6. Simpan pipeline lengkap + metadata dengan joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    statistik_input, contoh_sampel = hitung_statistik_input(X_train, y_train)

    bundle = {
        "pipeline": pipeline_terbaik,
        "fitur": FITUR_MENTAH,
        "fitur_rasio": FITUR_RASIO,
        "kelas": kelas,
        "nama_model_terbaik": nama_terbaik,
        "statistik_input": statistik_input,
        "contoh_sampel": contoh_sampel,
        "audit_data": hasil_audit,
        "korelasi_tinggi": korelasi_tinggi,
        "perbandingan_model": perbandingan_model,
        "metrik_cv": terbaik_info,
        "metrik_test": metrik_test,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report_dict,
        "versi_sklearn": sklearn.__version__,
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\n[6] Model & metadata berhasil disimpan ke: {MODEL_PATH.relative_to(BASE_DIR)}")

    # Simpan ringkasan JSON (tanpa objek pipeline) untuk keperluan pelaporan
    ringkasan_json = {k: v for k, v in bundle.items() if k != "pipeline"}
    SUMMARY_PATH.write_text(
        json.dumps(ringkasan_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    Ringkasan evaluasi JSON disimpan ke : {SUMMARY_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
