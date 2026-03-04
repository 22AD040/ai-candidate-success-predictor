import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.train_model import train_logistic
from models.train_model import train_knn
from models.train_model import train_linear
from utils.metrics import evaluate_model

st.set_page_config(page_title="AI Candidate Success Predictor",layout="wide")

menu = st.sidebar.selectbox("Menu",["Home","Dataset","Prediction","Live Prediction","Model Accuracy","Contact"])


if menu == "Home":

    st.title("AI Candidate Success Predictor")

    st.image("assets/home.jpg")

    st.markdown("### AI Powered Employee Attrition Prediction System")

    st.write("""
This AI system predicts whether an employee is likely to leave a company using machine learning.

The application analyzes employee data and evaluates multiple machine learning models to determine the most reliable predictions.

The dashboard provides interactive analytics and model performance insights.
""")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.info("Machine Learning Models")

    with c2:
        st.info("Prediction Analytics")

    with c3:
        st.info("Interactive Dashboard")



if menu == "Dataset":

    st.title("Dataset Analytics Dashboard")

    df = pd.read_csv("resume_dataset.csv")

    st.dataframe(df.head(20))

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Total Rows",df.shape[0])

    with col2:
        st.metric("Total Columns",df.shape[1])

    with col3:
        st.metric("Missing Values",df.isnull().sum().sum())



    st.subheader("Attrition Distribution")

    attrition_counts = df["Attrition"].value_counts()

    fig1,ax1 = plt.subplots(figsize=(5,3))   # SMALLER GRAPH

    ax1.bar(attrition_counts.index,attrition_counts.values)

    ax1.set_xlabel("Attrition")

    ax1.set_ylabel("Employees")

    st.pyplot(fig1)



    st.subheader("Correlation Heatmap")

    df_numeric = df.select_dtypes(include="number")

    fig2,ax2 = plt.subplots(figsize=(6,4))   # SMALLER GRAPH

    sns.heatmap(df_numeric.corr(),ax=ax2)

    st.pyplot(fig2)



    if "MonthlyIncome" in df.columns:

        st.subheader("Salary Distribution")

        fig3,ax3 = plt.subplots(figsize=(5,3))   # SMALLER GRAPH

        ax3.hist(df["MonthlyIncome"],bins=30)

        ax3.set_xlabel("Monthly Income")

        ax3.set_ylabel("Employees")

        st.pyplot(fig3)



if menu == "Prediction":

    st.title("Model Prediction Results")

    st.image("assets/prediction.jpg")

    model_choice = st.selectbox("Select Model",["Logistic Regression","KNN","Linear Regression"])

    if model_choice == "Logistic Regression":
        model,pred,y_test,X_train,y_train = train_logistic()

    if model_choice == "KNN":
        model,pred,y_test,X_train,y_train = train_knn()

    if model_choice == "Linear Regression":
        model,pred,y_test,X_train,y_train = train_linear()

    st.write("0 → Employee Stay")
    st.write("1 → Employee Leave")

    pred_df = pd.DataFrame(pred[:15],columns=["Prediction"])

    pred_df["Meaning"] = pred_df["Prediction"].map({0:"Employee Stay",1:"Employee Leave"})

    st.dataframe(pred_df)



if menu == "Live Prediction":

    st.markdown("<h1 style='text-align:center;'>AI Employee Attrition Prediction</h1>",unsafe_allow_html=True)

    st.markdown("""
    This intelligent HR analytics system predicts whether an employee is likely to leave the organization.
    
    The model analyzes employee attributes such as age, distance from home, education level, salary and years at company.
    
    The system provides probability scores to help HR teams identify potential attrition risks early.
    """)

    st.image("assets/prediction.jpg")

    df = pd.read_csv("resume_dataset.csv")

    df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

    df = df.select_dtypes(include="number")

    X = df.drop("Attrition",axis=1)

    model,pred,y_test,X_train,y_train = train_logistic()

    st.markdown("## Employee Information")

    col1,col2 = st.columns(2)

    with col1:
        age = st.slider("Age",18,60,30)
        distance = st.slider("Distance From Home",1,30,10)
        education = st.slider("Education Level",1,5,3)

    with col2:
        income = st.slider("Monthly Income",1000,20000,5000)
        years = st.slider("Years At Company",0,40,5)

    input_row = X.iloc[0].copy()

    input_row["Age"] = age
    input_row["DistanceFromHome"] = distance
    input_row["Education"] = education
    input_row["MonthlyIncome"] = income
    input_row["YearsAtCompany"] = years

    input_df = pd.DataFrame([input_row])

    predict = st.button("Predict Attrition Risk")

    if predict:

        prediction = model.predict(input_df)

        prob = model.predict_proba(input_df)

        leave_prob = prob[0][1]
        stay_prob = prob[0][0]

        st.markdown("## Prediction Result")

        col1,col2 = st.columns(2)

        with col1:
            st.metric("Stay Probability",str(round(stay_prob*100,2))+" %")

        with col2:
            st.metric("Leave Probability",str(round(leave_prob*100,2))+" %")

        fig,ax = plt.subplots(figsize=(4,3))   # SMALL GRAPH

        ax.bar(["Stay","Leave"],[stay_prob,leave_prob])

        ax.set_ylabel("Probability")

        st.pyplot(fig)

        if prediction[0] == 1:
            st.error("High Attrition Risk: Employee Likely To Leave")

        else:
            st.success("Low Attrition Risk: Employee Likely To Stay")




if menu == "Model Accuracy":

    st.title("Model Performance Dashboard")

    st.image("assets/accuracy.jpg")

    model,pred,y_test,X_train,y_train = train_logistic()

    accuracy,cm,report,mse,f1,cv = evaluate_model(model,X_train,y_train,y_test,pred)

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Accuracy",round(accuracy,3))

    with col2:
        st.metric("F1 Score",round(f1,3))   # FIXED

    with col3:
        st.metric("MSE",round(mse,3))



    st.subheader("Confusion Matrix")

    fig4,ax4 = plt.subplots(figsize=(4,3))   # SMALLER

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax4)

    ax4.set_xlabel("Predicted")

    ax4.set_ylabel("Actual")

    st.pyplot(fig4)



    st.subheader("Classification Report")

    st.text(report)



    st.subheader("Cross Validation Scores")

    fig5,ax5 = plt.subplots(figsize=(5,3))   # SMALLER

    ax5.bar(["Fold1","Fold2","Fold3","Fold4","Fold5"],cv)

    ax5.set_ylabel("Accuracy")

    st.pyplot(fig5)



    st.subheader("Prediction Distribution")

    pred_counts = pd.Series(pred).value_counts()

    fig6,ax6 = plt.subplots(figsize=(4,3))   # SMALLER

    ax6.bar(["Stay","Leave"],pred_counts)

    st.pyplot(fig6)




if menu == "Contact":

    st.title("Contact")

    st.image("assets/contact.jpg")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Developer")

        st.write("Ratchita B")

        st.subheader("Project")

        st.write("AI Candidate Success Predictor")

        st.subheader("Technology Stack")

        st.write("Python")
        st.write("Machine Learning")
        st.write("Scikit Learn")
        st.write("Streamlit")
        st.write("NumPy")
        st.write("Pandas")
        st.write("Matplotlib")

    with col2:

        st.subheader("Project Description")

        st.write("""
This AI powered system predicts employee attrition using machine learning models.

The system analyzes employee attributes and evaluates model performance using several machine learning metrics.

It provides interactive visualizations and prediction tools for HR analytics.
""")

        st.subheader("Key Features")

        st.write("Employee Attrition Prediction")
        st.write("Machine Learning Model Comparison")
        st.write("Cross Validation Analysis")
        st.write("Confusion Matrix Visualization")
        st.write("Interactive AI Dashboard")