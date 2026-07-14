import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# set page config to wide for more space
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# custom css to make the dashboard look better
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #2E75B6; }
    h1, h2, h3 { color: #333333; }
    .stButton>button { background-color: #2E75B6; color: white; border-radius: 8px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #1a5a94; color: white; }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# DATA AND MODEL LOADING
# ------------------------------------------------------------------

# cache data so it loads fast on reruns
@st.cache_data
def load_dataset():
    df = pd.read_csv("CustomerChurn.csv")
    
    # drop id, it's useless for analysis
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    
    # fix totalcharges empty strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # use assignment instead of inplace to avoid pandas warnings on cloud
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # map senior citizen back to text for readable charts
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    return df

# cache models so we don't load them every time
@st.cache_resource
def load_models_and_tools():
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

# preprocess data exactly like we did in the notebook
def preprocess_new_data(input_df, scaler, feature_columns):
    data = input_df.copy()
    
    # handle totalcharges just in case
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    
    # map senior citizen back to text if it's 0/1
    if set(data["SeniorCitizen"].unique()).issubset({0, 1}):
        data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
        
    # drop target if it's there
    if "Churn" in data.columns:
        data.drop(columns=["Churn"], inplace=True)
        
    # step 1: label encode binary columns exactly like the notebook
    # manual mapping is safer than labelencoder for new data
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
    mapping = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map(mapping)
            
    # step 2: one-hot encode multi-category columns
    multi_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                  "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)
    
    # step 3: make sure all training columns exist
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
            
    # step 4: put columns in the exact same order
    data = data[feature_columns]
    
    # step 5: scale the numbers
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # bulletproof: fill any remaining NaNs with 0 so cloud server never crashes
    data = data.fillna(0)
    
    return data


# ------------------------------------------------------------------
# PAGE LAYOUT FUNCTIONS
# ------------------------------------------------------------------

def show_overview(df):
    st.markdown("<h1>Dashboard Overview</h1>", unsafe_allow_html=True)
    st.write("A quick snapshot of customer churn in the dataset.")
    
    # basic math for metrics
    total_customers = len(df)
    churned_customers = len(df[df["Churn"] == "Yes"])
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly = df["MonthlyCharges"].mean()

    # show metrics in 4 columns
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
        # donut chart looks nicer than standard pie
        fig_donut = px.pie(
            churn_counts, values="Count", names="Status", hole=0.5,
            color="Status", color_discrete_map={"No": "#2E75B6", "Yes": "#C00000"}
        )
        # clean up chart layout
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), template="plotly_white")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig_bar = px.bar(
            contract_churn, x="Contract", y="Churn Rate (%)", 
            color="Contract", color_discrete_sequence=["#C00000"]
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)


def show_churn_analysis(df):
    st.markdown("<h1>Churn Analysis</h1>", unsafe_allow_html=True)
    st.write("Understanding the patterns behind why customers leave.")
    
    # need numeric churn for grouping
    df["ChurnNumeric"] = df["Churn"].map({"Yes": 1, "No": 0})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Tenure")
        tenure_churn = df.groupby("tenure")["ChurnNumeric"].mean().reset_index()
        fig_tenure = px.line(tenure_churn, x="tenure", y="ChurnNumeric", markers=True)
        fig_tenure.update_traces(line_color="#C00000")
        fig_tenure.update_layout(
            xaxis_title="Tenure (Months)", yaxis_title="Churn Rate", 
            margin=dict(t=10, b=10, l=10, r=10), template="plotly_white"
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        st.subheader("Monthly Charges Distribution")
        fig_hist = px.histogram(
            df, x="MonthlyCharges", color="Churn", marginal="box", 
            color_discrete_map={"No": "#2E75B6", "Yes": "#C00000"}
        )
        fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10), barmode="overlay", template="plotly_white")
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn by Internet Service")
        internet_churn = df.groupby("InternetService")["ChurnNumeric"].mean().reset_index()
        internet_churn["ChurnNumeric"] = internet_churn["ChurnNumeric"] * 100
        fig_internet = px.bar(
            internet_churn, x="InternetService", y="ChurnNumeric", 
            color="InternetService", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_internet.update_layout(
            showlegend=False, xaxis_title="Internet Service", yaxis_title="Churn Rate (%)", 
            margin=dict(t=10, b=10, l=10, r=10), template="plotly_white"
        )
        st.plotly_chart(fig_internet, use_container_width=True)

    with col4:
        st.subheader("Churn by Payment Method")
        payment_churn = df.groupby("PaymentMethod")["ChurnNumeric"].mean().reset_index()
        payment_churn["ChurnNumeric"] = payment_churn["ChurnNumeric"] * 100
        fig_payment = px.bar(
            payment_churn, x="PaymentMethod", y="ChurnNumeric", 
            color="PaymentMethod", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_payment.update_layout(
            showlegend=False, xaxis_title="Payment Method", yaxis_title="Churn Rate (%)", 
            margin=dict(t=10, b=10, l=10, r=10), template="plotly_white"
        )
        st.plotly_chart(fig_payment, use_container_width=True)


def show_model_performance(df, models_dict, scaler, feature_columns):
    st.markdown("<h1>Model Performance</h1>", unsafe_allow_html=True)
    st.write("Comparing the three models trained on the data. Recall is our priority metric.")

    if models_dict is None:
        st.warning("Models not found. Please train and save your models first.")
        return

    # recreate test set to evaluate live
    X = df.drop(columns=["Churn", "ChurnNumeric"], errors="ignore").copy()
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    X_processed = preprocess_new_data(X, scaler, feature_columns)
    _, X_test, _, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    # loop through models and plot their confusion matrices
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)

        # save metrics for the table later
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Precision": precision_score(y_test, y_pred) * 100,
            "Recall": recall_score(y_test, y_pred) * 100,
            "F1-Score": f1_score(y_test, y_pred) * 100
        }

        with cols[idx]:
            st.subheader(name)
            cm = confusion_matrix(y_test, y_pred)
            # use plotly heatmap for a nicer confusion matrix
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig_cm.update_layout(
                xaxis_title="Predicted", yaxis_title="Actual",
                xaxis=dict(tickvals=[0, 1], ticktext=["No Churn", "Churn"]),
                yaxis=dict(tickvals=[0, 1], ticktext=["No Churn", "Churn"]),
                margin=dict(t=30, b=10, l=10, r=10), template="plotly_white"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparison Table")
    results_df = pd.DataFrame(results).T
    # highlight the best score in each column
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)


def show_single_prediction(best_model, scaler, feature_columns):
    st.markdown("<h1>Predict Single Customer</h1>", unsafe_allow_html=True)
    st.write("Fill in the details to see the churn risk for one customer.")

    if best_model is None:
        st.warning("Best model not found. Please train and save your models first.")
        return

    # using st.form so it only runs when the button is clicked
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
            # pack user inputs into a dict
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

            # convert to dataframe and process
            input_df = pd.DataFrame([user_input])
            processed_input = preprocess_new_data(input_df, scaler, feature_columns)
            
            # get prediction and probability
            prediction = best_model.predict(processed_input)
            probability = best_model.predict_proba(processed_input)[0]

            st.markdown("---")
            # show result with color coding
            if prediction[0] == 1:
                st.error(f"Prediction: This customer is likely to CHURN. (Probability: {probability[1]*100:.1f}%)")
            else:
                st.success(f"Prediction: This customer is likely to STAY. (Probability: {probability[0]*100:.1f}%)")


def show_batch_prediction(best_model, scaler, feature_columns):
    st.markdown("<h1>Batch Predictions</h1>", unsafe_allow_html=True)
    st.write("Upload a CSV file to get churn predictions for multiple customers.")

    if best_model is None:
        st.warning("Best model not found. Please train and save your models first.")
        return

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        # process the whole batch
        processed_batch = preprocess_new_data(batch_df.copy(), scaler, feature_columns)

        # get predictions
        predictions = best_model.predict(processed_batch)
        probabilities = best_model.predict_proba(processed_batch)[:, 1]

        # add results to original data
        result_df = batch_df.copy()
        result_df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
        result_df["Churn_Probability (%)"] = np.round(probabilities * 100, 1)

        st.markdown("---")
        st.subheader("Predictions Results")
        st.dataframe(result_df, use_container_width=True)

        # let user download the results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )


# ------------------------------------------------------------------
# MAIN APP LOGIC
# ------------------------------------------------------------------

def main():
    # load everything first
    df = load_dataset()
    best_model, models_dict, scaler, feature_columns = load_models_and_tools()

    # sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview", 
        "Churn Analysis", 
        "Model Performance", 
        "Predict Single Customer", 
        "Batch Predictions"
    ])

    # simple routing
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