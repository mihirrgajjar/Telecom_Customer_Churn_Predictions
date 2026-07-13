import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

@st.cache_data
def load_dataset():
    """
    Load the dataset and apply the exact same cleaning steps from Phase 2.
    We do this here so the dashboard charts display the correct, cleaned data.
    """
    df = pd.read_csv("CustomerChurn.csv")
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df

@st.cache_resource
def load_models_and_tools():
    """
    Load all saved models, the scaler, and the feature columns.
    We wrap this in a try-except block so the dashboard doesn't crash 
    if the user hasn't trained the models yet.
    """
    if not os.path.exists("models"):
        return None, None, None, None
        
    try:
        best_model = joblib.load("models/best_model.pkl")
        lr_model = joblib.load("models/lr_model.pkl")
        dt_model = joblib.load("models/dt_model.pkl")
        rf_model = joblib.load("models/rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        
        models = {
            "Logistic Regression": lr_model,
            "Decision Tree": dt_model,
            "Random Forest": rf_model
        }
        
        return best_model, models, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def preprocess_new_data(input_df, scaler, feature_columns):
    """
    This function takes raw user input (single or batch) and applies the 
    exact same preprocessing steps used during Phase 4 of the notebook.
    """
    data = input_df.copy()

    # User uploads can carry the same blank TotalCharges values as the raw file.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    if set(data["SeniorCitizen"].unique()).issubset({0, 1}):
        data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if "Churn" in data.columns:
        data.drop(columns=["Churn"], inplace=True)

    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", 
        "SeniorCitizen", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaymentMethod"
    ]

    # Match the notebook preprocessing so the saved models see familiar columns.
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    return data


def show_overview(df):
    st.header("Overview")
    st.write("A quick snapshot of customer churn in the dataset.")

    # These headline numbers give the dashboard its quick first read.
    total_customers = len(df)
    churned_customers = len(df[df["Churn"] == "Yes"])
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly = df["MonthlyCharges"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            churn_counts, values="Count", names="Status", hole=0.4,
            color="Status", color_discrete_map={"No": "#2E75B6", "Yes": "#C00000"}
        )
        fig_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig_bar = px.bar(
            contract_churn, x="Contract", y="Churn Rate (%)", 
            color="Contract", color_discrete_sequence=["#C00000"]
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)


def show_churn_analysis(df):
    st.header("Churn Analysis")
    st.write("Understanding the patterns behind why customers leave.")

    # A numeric churn flag makes the grouped rate charts straightforward.
    df["ChurnNumeric"] = df["Churn"].map({"Yes": 1, "No": 0})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Tenure")
        tenure_churn = df.groupby("tenure")["ChurnNumeric"].mean().reset_index()
        fig_tenure = px.line(tenure_churn, x="tenure", y="ChurnNumeric", markers=True)
        fig_tenure.update_layout(
            xaxis_title="Tenure (Months)", yaxis_title="Churn Rate", 
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        st.subheader("Monthly Charges Distribution")
        fig_hist = px.histogram(
            df, x="MonthlyCharges", color="Churn", marginal="box", 
            color_discrete_map={"No": "#2E75B6", "Yes": "#C00000"}
        )
        fig_hist.update_layout(margin=dict(t=0, b=0, l=0, r=0), barmode="overlay")
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn by Internet Service")
        internet_churn = df.groupby("InternetService")["ChurnNumeric"].mean().reset_index()
        internet_churn["ChurnNumeric"] = internet_churn["ChurnNumeric"] * 100
        fig_internet = px.bar(
            internet_churn, x="InternetService", y="ChurnNumeric", 
            color="InternetService"
        )
        fig_internet.update_layout(
            showlegend=False, xaxis_title="Internet Service", yaxis_title="Churn Rate (%)", 
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_internet, use_container_width=True)

    with col4:
        st.subheader("Churn by Payment Method")
        payment_churn = df.groupby("PaymentMethod")["ChurnNumeric"].mean().reset_index()
        payment_churn["ChurnNumeric"] = payment_churn["ChurnNumeric"] * 100
        fig_payment = px.bar(
            payment_churn, x="PaymentMethod", y="ChurnNumeric", 
            color="PaymentMethod"
        )
        fig_payment.update_layout(
            showlegend=False, xaxis_title="Payment Method", yaxis_title="Churn Rate (%)", 
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_payment, use_container_width=True)


def show_model_performance(df, models_dict, scaler, feature_columns):
    st.header("Model Performance")
    st.write("Comparing the three models trained on the data. Recall is our priority metric.")

    if models_dict is None:
        st.warning("Models not found. Please train and save your models first.")
        return

    X = df.drop(columns=["Churn", "ChurnNumeric"], errors="ignore").copy()
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    X_processed = preprocess_new_data(X, scaler, feature_columns)
    
    _, X_test, _, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

        with cols[idx]:
            st.subheader(name)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig_cm.update_layout(
                xaxis_title="Predicted", yaxis_title="Actual",
                xaxis=dict(tickvals=[0, 1], ticktext=["No Churn", "Churn"]),
                yaxis=dict(tickvals=[0, 1], ticktext=["No Churn", "Churn"]),
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparison Table")
    results_df = pd.DataFrame(results).T
    # Let the strongest metric in each column stand out at a glance.
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)


def show_single_prediction(best_model, scaler, feature_columns):
    st.header("Predict Churn for a Single Customer")
    st.write("Fill in the customer details below to see their churn risk.")

    if best_model is None:
        st.warning("Best model not found. Please train and save your models first.")
        return

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        col4, col5, col6 = st.columns(3)

        with col4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col5:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges", 18.0, 118.0, 50.0)

        with col6:
            total_charges = st.slider("Total Charges", 18.0, 9000.0, 1000.0)

        submit = st.form_submit_button("Predict Churn")

        if submit:
            user_input = {
                "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet_service,
                "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                "DeviceProtection": device_protection, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            # Keep the manual entry path identical to batch prediction.
            input_df = pd.DataFrame([user_input])
            processed_input = preprocess_new_data(input_df, scaler, feature_columns)
            
            prediction = best_model.predict(processed_input)
            probability = best_model.predict_proba(processed_input)[0]

            st.markdown("---")
            if prediction[0] == 1:
                st.error(f"Prediction: This customer is likely to CHURN. (Probability: {probability[1]*100:.1f}%)")
            else:
                st.success(f"Prediction: This customer is likely to STAY. (Probability: {probability[0]*100:.1f}%)")


def show_batch_prediction(best_model, scaler, feature_columns):
    st.header("Batch Predictions")
    st.write("Upload a CSV file containing customer data to get churn predictions for multiple customers at once.")

    if best_model is None:
        st.warning("Best model not found. Please train and save your models first.")
        return

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        # Predict from a cleaned copy, then attach results to the user's original rows.
        processed_batch = preprocess_new_data(batch_df.copy(), scaler, feature_columns)

        predictions = best_model.predict(processed_batch)
        probabilities = best_model.predict_proba(processed_batch)[:, 1]

        result_df = batch_df.copy()
        result_df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
        result_df["Churn_Probability (%)"] = np.round(probabilities * 100, 1)

        st.markdown("---")
        st.subheader("Predictions Results")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

def main():
    df = load_dataset()
    best_model, models_dict, scaler, feature_columns = load_models_and_tools()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", [
        "Overview", 
        "Churn Analysis", 
        "Model Performance", 
        "Predict Single Customer", 
        "Batch Predictions"
    ])

    if page == "Overview":
        show_overview(df)
    elif page == "Churn Analysis":
        show_churn_analysis(df)
    elif page == "Model Performance":
        show_model_performance(df, models_dict, scaler, feature_columns)
    elif page == "Predict Single Customer":
        show_single_prediction(best_model, scaler, feature_columns)
    elif page == "Batch Predictions":
        show_batch_prediction(best_model, scaler, feature_columns)

if __name__ == "__main__":
    main()