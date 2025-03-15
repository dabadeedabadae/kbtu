import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Кэширование загрузки данных
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# Кэширование кодирования категориальных признаков
@st.cache_data
def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Кэширование обучения модели
@st.cache_resource
def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1  # Используем все доступные ядра
    )
    model.fit(X_train, y_train)
    return model


st.title("🚀 Оптимизированное обучение XGBoost на 70% данных")

uploaded_file = st.file_uploader("Загрузите CSV или Excel-файл", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Первые 5 строк данных:")
    st.dataframe(df.head())

    target_column = st.selectbox("Выберите целевую переменную (таргет)", df.columns, index=None,
    placeholder="Таргет не выбран")

    if target_column is not None:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X, label_encoders = encode_categorical(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        y_train = LabelEncoder().fit_transform(y_train)

        st.write(f"**Размер тренировочной выборки:** {X_train.shape[0]} строк")
        st.write(f"**Размер тестовой выборки:** {X_test.shape[0]} строк")

        # Обучение модели
        start_time = time.time()
        model = train_xgboost(X_train, y_train)
        train_time = time.time() - start_time
        st.write(f"✅ **Модель обучена за {train_time:.2f} секунд**")

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        st.write("### 🔥 Результаты модели:")
        st.write(f"✅ **Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"✅ **Precision:** {precision_score(y_test, y_pred, average='weighted'):.4f}")
        st.write(f"✅ **Recall:** {recall_score(y_test, y_pred, average='weighted'):.4f}")
        st.write(f"✅ **ROC AUC:** {roc_auc_score(y_test, y_pred_proba):.4f}")

        # График важности признаков
        st.write("### Важность признаков")
        feature_importances = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 177))
        sns.barplot(x="Importance", y="Feature", data=feature_importances, ax=ax)
        st.pyplot(fig)

        feature_importances.reset_index(drop=True, inplace=True)
        st.dataframe(feature_importances)

        st.success("✅ Обучение завершено!")

        # Ввод данных вручную
        st.write("### 🔍 Введите данные для предсказания вероятности выплаты кредита")
        input_data = {}
        for col in X.columns:
            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                selected = st.selectbox(f"{col}", options)
                input_data[col] = label_encoders[col].transform([selected])[0]
            else:
                input_data[col] = st.number_input(f"{col}", value=float(X[col].median()))

        if st.button("Предсказать вероятность выплаты кредита"):
            input_df = pd.DataFrame([input_data])
            probability = model.predict_proba(input_df)[:, 1][0]
            st.write(f"🔮 **Вероятность выплаты кредита:** {probability:.4f}")
