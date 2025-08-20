
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tensorflow.keras.models import load_model as keras_load_model

# -------------------
# Load tools
# -------------------
scaler = joblib.load("scaler.pkl")              # Scaler used during training
feature_columns = joblib.load("feature_columns.pkl")  # Feature names used

# -------------------
# Streamlit page setup
# -------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title(" Predictive Maintenance for Industrial Machinery")
st.markdown("Choose either **manual input**, **file upload**, or **model evaluation**.")

# -------------------
# Model loader
# -------------------
def load_model(model_choice):
    if model_choice == "Random Forest":
        return joblib.load("rf_model.pkl")
    elif model_choice == "XGBoost":
        return joblib.load("xgboost_model.pkl")
    elif model_choice == "LSTM":
        return keras_load_model("lstm_model.h5")

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3 = st.tabs([" Manual Input", " Upload File", " Model Evaluation"])

# -------------------
# Tab 1: Manual Input
# -------------------
with tab1:
    st.subheader("Manual Input of Sensor Values")
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"], key="manual_model")
    model = load_model(model_choice)

    # User input
    user_input = {col: st.number_input(f"Enter {col}", value=0.0, key=f"{col}_manual") for col in feature_columns}
    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)

    # Reshape for LSTM
    if model_choice == "LSTM":
        scaled_input = np.expand_dims(scaled_input, axis=1)

    if st.button("Predict Machine Failure (Manual Input)"):
        y_pred = model.predict(scaled_input)

        if model_choice == "LSTM":
            proba = y_pred.ravel()[0]
            prediction = int(proba > 0.5)
        else:
            try:
                proba = model.predict_proba(scaled_input)[0][1]
            except:
                proba = y_pred.ravel()[0]
            prediction = int(model.predict(scaled_input)[0])

        if prediction == 1:
            st.error(f" Machine is likely to fail! (Risk Score: {proba:.2f})")
        else:
            st.success(f" Machine is healthy (Risk Score: {proba:.2f})")

# -------------------
# Tab 2: Batch Upload
# -------------------
with tab2:
    st.subheader(" Batch Prediction")
    batch_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"], key="batch_upload")
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"], key="batch_model")
    timesteps = st.slider("Select timesteps (LSTM only)", 5, 30, 10, key="batch_timesteps")

    if batch_file is not None:
        if batch_file.name.endswith(".csv"):
            data = pd.read_csv(batch_file)
        else:
            data = pd.read_excel(batch_file)

        st.dataframe(data.head())
        if "Type" in data.columns:
            data = pd.get_dummies(data, columns=["Type"])

        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0

        X_data = data[feature_columns]
        X_scaled = scaler.transform(X_data)

        def create_sequences(X, timesteps=10):
            Xs = []
            for i in range(len(X) - timesteps):
                Xs.append(X[i:i+timesteps])
            return np.array(Xs)

        if model_choice == "LSTM":
            model = load_model("LSTM")
            X_seq = create_sequences(X_scaled, timesteps)
            y_proba = model.predict(X_seq).ravel()
            y_pred = (y_proba >= 0.5).astype(int)
            result_df = data.iloc[timesteps:].copy()
            result_df["Failure_Prediction"] = y_pred
            result_df["Failure_Probability"] = y_proba
        else:
            model = load_model(model_choice)
            try:
                y_proba = model.predict_proba(X_scaled)[:, 1]
            except:
                y_proba = model.predict(X_scaled).ravel()
            y_pred = model.predict(X_scaled)
            result_df = data.copy()
            result_df["Failure_Prediction"] = y_pred
            result_df["Failure_Probability"] = y_proba

        st.dataframe(result_df.head(20))
        failures = int((result_df["Failure_Prediction"] == 1).sum())
        healthy = int((result_df["Failure_Prediction"] == 0).sum())
        total = len(result_df)
        st.info(f"Total: {total} | Healthy: {healthy} | Predicted Failures: {failures}")

        fig, ax = plt.subplots()
        ax.pie([healthy, failures], labels=["Healthy", "Failures"], autopct="%1.1f%%",
               colors=["#4CAF50", "#F44336"])
        ax.set_title(f"Machine Health Distribution ({model_choice})")
        st.pyplot(fig)

        csv_download = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions as CSV", csv_download, "batch_predictions.csv", "text/csv")


# -------------------
# Tab 3: Model Evaluation
# -------------------
with tab3:
    st.subheader(" Model Evaluation / Prediction")
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"], key="eval_model")
    eval_file = st.file_uploader("Upload test CSV/Excel", type=["csv", "xlsx"], key="eval_upload")
    timesteps = st.slider("Select timesteps (LSTM only)", 5, 30, 10, key="eval_timesteps")

    if eval_file is not None:
        if eval_file.name.endswith(".csv"):
            test_data = pd.read_csv(eval_file)
        else:
            test_data = pd.read_excel(eval_file)

        st.dataframe(test_data.head())
        if "Type" in test_data.columns:
            test_data = pd.get_dummies(test_data, columns=["Type"])
        for col in feature_columns:
            if col not in test_data.columns:
                test_data[col] = 0

        X_scaled = scaler.transform(test_data[feature_columns])

        def create_sequences(X, y=None, timesteps=10):
            Xs, ys = [], []
            for i in range(len(X) - timesteps):
                Xs.append(X[i:i+timesteps])
                if y is not None:
                    ys.append(y[i+timesteps])
            return np.array(Xs), (np.array(ys) if y is not None else None)

        if model_choice == "LSTM":
            model = load_model("LSTM")
            if "Machine failure" in test_data.columns:
                y_true = test_data["Machine failure"].values
                X_seq, y_seq = create_sequences(X_scaled, y_true, timesteps)
                y_proba = model.predict(X_seq).ravel()
                y_pred = (y_proba >= 0.5).astype(int)
                y_test = y_seq
            else:
                X_seq, _ = create_sequences(X_scaled, None, timesteps)
                y_proba = model.predict(X_seq).ravel()
                y_pred = (y_proba >= 0.5).astype(int)
                result_df = test_data.iloc[timesteps:].copy()
                result_df["Failure_Prediction"] = y_pred
                result_df["Failure_Probability"] = y_proba
                st.dataframe(result_df.head(20))
        else:
            model = load_model(model_choice)
            try:
                y_proba = model.predict_proba(X_scaled)[:, 1]
            except:
                y_proba = model.predict(X_scaled).ravel()
            y_pred = model.predict(X_scaled)
            if "Machine failure" in test_data.columns:
                y_test = test_data["Machine failure"]
            else:
                result_df = test_data.copy()
                result_df["Failure_Prediction"] = y_pred
                result_df["Failure_Probability"] = y_proba
                st.dataframe(result_df.head(20))

        # ADD RESULT SUMMARY HERE
        st.write("### Prediction Summary")
        healthy = int((y_pred == 0).sum())
        failures = int((y_pred == 1).sum())
        total = len(y_pred)
        st.info(f"Total: {total} | Healthy (0): {healthy} | Failures (1): {failures}")

        if "Machine failure" in test_data.columns:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

