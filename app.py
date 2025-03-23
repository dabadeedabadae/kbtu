import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

st.markdown("""
    <style>
    /* Сайдбар фон */
    section[data-testid="stSidebar"] {
        background-color: #2D5084 !important;
    }

    /* Контейнер selectbox */
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
        margin-top: 10px;
    }

    /* Надпись "Выберите вкладку" */
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .css-1c7y2kd,   /* для старых версий Streamlit */
    div[data-testid="stSidebar"] .css-10trblm,   /* для новых версий Streamlit */
    div[data-testid="stSidebar"] .e1f1d6gn4,     /* ещё один вариант класса */
    div[data-testid="stSidebar"] .st-bv {
        color: white !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }

    /* Селектбокс (выбранное значение) */
    div[data-testid="stSidebar"] div[role="combobox"] {
        background-color: #001E48 !important;
        border: 1px solid #001E48 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        color: white !important;
    }

    /* Внутренности */
    div[data-testid="stSidebar"] div[role="combobox"] input,
    div[data-testid="stSidebar"] div[role="combobox"] span {
        background-color: #001E48 !important;
        color: white !important;
    }

    div[data-testid="stSidebar"] div[role="combobox"] svg {
        color: white !important;
    }

    /* Выпадающий список */
    ul[role="listbox"] {
        background-color: #001E48 !important;
        color: white !important;
        border: none !important;
    }

    li[role="option"] {
        background-color: #001E48 !important;
        color: white !important;
    }

    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: #003366 !important;
    }

    /* BaseWeb select styles override */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        background-color: #001E48 !important;
        color: white !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)





# Пути для сохранения моделей и параметров
MODEL_PATH_XGB = "model_xgb.json"
MODEL_PATH_LR = "model_lr.json"
LABEL_ENCODERS_PATH = "label_encoders.json"
FEATURES_PATH = "features.json"
SCALER_PATH = "scaler.json"

# Выбор вкладки через selectbox
menu = st.sidebar.selectbox(
    "Выберите вкладку",
    [
        "🏋️‍♂️ Обучение XGBoost",
        "📊 Обучение Logistic Regression",
        "🔍 Проверка XGBoost",
        "📊 Проверка Logistic Regression"
    ]
)

# Выбор типа модели
if "XGBoost" in menu:
    model_type = "XGBoost" if "Обучение" in menu else "xgb"
elif "Logistic Regression" in menu:
    model_type = "Logistic Regression" if "Обучение" in menu else "lr"

# Функции

def save_model(model, feature_names, model_path):
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(model_path)
    else:
        with open(model_path, "w") as f:
            json.dump({"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "features": feature_names}, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)

def load_model(model_path, model_type):
    if model_type == "xgb":
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        with open(model_path, "r") as f:
            data = json.load(f)
        model = LogisticRegression()
        model.coef_ = np.array(data["coef"])
        model.intercept_ = np.array(data["intercept"])
        model.classes_ = np.array([0, 1])
    return model

def save_label_encoders(label_encoders):
    with open(LABEL_ENCODERS_PATH, "w") as f:
        json.dump({col: le.classes_.tolist() for col, le in label_encoders.items()}, f)

def load_label_encoders():
    with open(LABEL_ENCODERS_PATH, "r") as f:
        return {col: LabelEncoder().fit(classes) for col, classes in json.load(f).items()}

def save_scaler(scaler):
    with open(SCALER_PATH, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

def load_scaler():
    with open(SCALER_PATH, "r") as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.scale_ = np.array(data["scale"])
    return scaler

def load_feature_names():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)

def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def align_features(df, feature_names):
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# Интерфейс
st.title("🚀 Антифрод ML")

if "Обучение" in menu:
    st.header(f"📌 Обучение модели {model_type}")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel-файл", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        if "GB_flag" not in df.columns:
            st.error("❌ Ошибка: В файле нет столбца 'GB_flag'")
        else:
            X, y = df.drop(columns=["GB_flag"]), df["GB_flag"]
            X, label_encoders = encode_categorical(X)
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            feature_names = list(X.columns)
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="logloss", n_jobs=-1) if model_type == "XGBoost" else LogisticRegression()
            model.fit(X_train, y_train)
            model_path = MODEL_PATH_XGB if model_type == "XGBoost" else MODEL_PATH_LR
            save_model(model, feature_names, model_path)
            save_label_encoders(label_encoders)
            save_scaler(scaler)
            st.success(f"✅ {model_type} модель сохранена")

elif "Проверка" in menu:
    model_path = MODEL_PATH_XGB if model_type == "xgb" else MODEL_PATH_LR
    if os.path.exists(model_path):
        model = load_model(model_path, model_type)
        scaler = load_scaler()
        feature_names = load_feature_names()
        uploaded_file = st.file_uploader("Загрузите тестовый файл", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            df_original = df.copy()
            df = align_features(df, feature_names)
            imputer = SimpleImputer(strategy="mean")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)
            predictions = model.predict_proba(df)[:, 1]
            df_original["Вероятность мошенничества"] = predictions
            st.dataframe(df_original)
    else:
        st.error("❌ Сначала обучите модель")
