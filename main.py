import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import time

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("Employee Attrition Prediction using ML")

@st.cache_data
def load_data():
    df = pd.read_csv("employee_data.csv")
    return df

df = load_data()
df_left = df[df["Attrition"]=="Yes"]
df_stayed = df[df["Attrition"]=="No"]

@st.cache_data
def clear_data():
    dfml = df
    dfml["Attrition"] = dfml["Attrition"].apply(lambda x:1 if x =='Yes' else 0)
    dfml["Over18"] = dfml["Over18"].apply(lambda x:1 if x =='Y' else 0)
    dfml["OverTime"] = dfml["OverTime"].apply(lambda x:1 if x =='Yes' else 0)
    dfml.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'], axis=1, inplace=True)
    return dfml

# Machine Learning data preprocessing
dfml = clear_data()
X_cat = dfml[['BusinessTravel', 'Department','EducationField','Gender','JobRole','MaritalStatus',]]
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
X_num = dfml.drop(columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus','Attrition'])
X_all = pd.concat([X_cat, X_num], axis=1)
X_all.columns = X_all.columns.astype(str)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)
y = dfml["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

opt = st.sidebar.radio("Please select one option below:?", ("Introduction","Explonatory Data Analysis (EDA)","Data Visualization","Predictive Analysis"))

if opt=="Introduction":
    st.header("Introduction")
    st.subheader("Project Overview.")
    st.write("Employee attrition is a significant challenge for many organizations, as the loss of skilled and experienced employees can have a substantial impact on productivity, team dynamics, and overall business performance. Predicting employee attrition is crucial for businesses to proactively identify and address the factors that contribute to employee turnover, allowing them to implement effective retention strategies.")
    st.write("The goal of this project is to develop a machine learning model that can accurately predict employee attrition based on various factors, such as job satisfaction, work-life balance, compensation, and career development opportunities. By understanding the key drivers of employee attrition, organizations can make informed decisions to improve employee engagement, job satisfaction, and overall workforce retention.")
    image = Image.open('image.jpg')
    # Display the image
    st.image(image, caption='Low Atrrition Boost Retention', use_column_width=True)
    st.subheader("Objectives")
    st.markdown("""
    - Data Collection and Preprocessing: Gather relevant employee data, including demographic information, job-related factors, and performance metrics. Clean and preprocess the data to ensure it is suitable for model training and analysis.
    - Exploratory Data Analysis (EDA): Conduct a thorough EDA to identify key factors that influence employee attrition. Analyze the relationships between different variables and the target variable (employee attrition).
    - Feature Engineering: Based on the insights gained from the EDA, engineer new features that may improve the model's predictive accuracy. This may include creating derived variables, handling missing values, and addressing any data quality issues.
    - Model Development and Evaluation: Select appropriate machine learning algorithms, such as logistic regression, decision trees, random forests, or gradient boosting, and train models to predict employee attrition. Evaluate the models using various performance metrics, such as accuracy, precision, recall, and F1-score.
    - Model Optimization and Hyperparameter Tuning: Fine-tune the selected models by optimizing their hyperparameters to further improve their predictive performance. Use techniques like grid search or random search to find the optimal hyperparameter values.
    - Deployment and Monitoring: Integrate the best-performing model into the organization's HR systems or decision-making processes. Continuously monitor the model's performance and update it as new data becomes available to ensure its accuracy and relevance over time.
    """)
    st.subheader("Expected Outcome")
    st.markdown("""
    - A robust machine learning model that can accurately predict employee attrition based on the provided data.
    - Insights into the key factors that influence employee attrition, which can guide the organization's retention strategies.
    - A framework for proactively identifying employees at risk of leaving, enabling the HR team to implement targeted interventions and improve overall workforce retention.
    - A scalable and maintainable solution that can be integrated into the organization's HR systems and decision-making processes.
    """)



elif opt=="Explonatory Data Analysis (EDA)":
    st.header("Explonatory Data Analysis (EDA)")
    st.write("The employee attrition dataset is a popular machine learning dataset that provides information about employee turnover at a company. The dataset contains information on various factors that may influence an employee's decision to leave the organization.")
    st.write("The page will provide the explonatory data anlysis (EDA) on the dataset. The overall dataset shapes, info and attributes. In overall, the dataset is clean as there is no missing value, all the data was entered corretly and data type is clear and easy to manage.")
    if st.checkbox("Show raw data", True):
        st.markdown("Table below shows the raw data of employee")
        st.write(df.head(10))
    st.write("Dataset Shape - # of Rows and Columns")
    with st.expander("Show Shape"):
        st.text("Number of rows in this dataset is {}".format(df.shape[0]))
        st.text("Number of columns in this dataset is {}".format(df.shape[1]))
    st.write("Dataset Information - there is no missing data in this dataset, 26 x int64 and 9 objects")
    with st.expander("Show Info"):
        # Capture the output of df.info()
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    st.write("Dataset Statistical Summary of Each Columns")
    with st.expander("Show summary"):
        st.table(df.describe(include='all'))

elif opt =="Data Visualization":
    st.header("Data visualization")
    st.write("Data visualization is a crucial tool that improves understanding, communication, and decision-making by transforming complex data into clear, concise, and compelling visual representations. It helps uncover hidden insights, identify data quality issues, and enables interactive exploration, fostering a data-driven approach to problem-solving. Visualizations effectively convey findings to stakeholders, support better-informed decisions, and allow users to craft engaging narratives with data. By leveraging the power of data visualization, organizations can gain deeper insights, enhance collaboration, and make more informed strategic choices.")
    if not st.checkbox("Hide Histogram Data", True, key='histogram'):
        st.write("Histogram of the attributes")
        #fig, ax = plt.subplots()
        # Create the progress bar
        progress_bar = st.progress(0)
        # Display the progress label
        progress_label = st.empty()
        progress_label.text("Loading Histogram...")
        dfml.hist(bins=30, figsize=(20,20), color='blue')
        progress_bar.progress(100)
        st.pyplot()
    if not st.checkbox("Hide Correlation Chart", True, key='correlation'):
        st.write("Correlation of attributes")
        dfml_numeric = dfml.select_dtypes(include=['number'])
        corr = dfml_numeric.corr()
        f, ax = plt.subplots(figsize = (20,20))
        sns.heatmap(corr, annot = True)
        st.pyplot()
    if not st.checkbox("Hide Attributes Analysis", True, key='jobrole'):
        plt.figure(figsize=(20,20))
        plt.subplot(411)
        sns.countplot(x='JobRole', hue='Attrition', data=df)
        plt.subplot(412)
        sns.countplot(x='MaritalStatus', hue='Attrition', data=df)
        plt.subplot(413)
        sns.countplot(x='JobInvolvement', hue='Attrition', data=df)
        plt.subplot(414)
        sns.countplot(x='JobLevel', hue='Attrition', data=df)
        st.pyplot()
    if not st.checkbox("Hide Monthly Income vs Job Role Anlaysis", True, key='monthlyincome'):
        sns.boxplot(x='MonthlyIncome', y='JobRole', data=df)
        st.pyplot()
    if not st.checkbox("Hide KDE (Kernel Density Estimate) - Distance From Home", True, key='kde'):
        plt.figure(figsize=(12,7))
        sns.kdeplot(df_left['DistanceFromHome'], label='Employee Who Left', shade=True, color='r')
        sns.kdeplot(df_stayed['DistanceFromHome'], label='Employee Who Stayed', shade=True, color='b')
        plt.legend()
        st.pyplot()
    if not st.checkbox("Hide KDE (Kernel Density Estimate) - Years with Current Manager", True, key='YWCM'):
        plt.figure(figsize=(12,7))
        sns.kdeplot(df_left['YearsWithCurrManager'], label='Employee Who Left', shade=True, color='r')
        sns.kdeplot(df_stayed['YearsWithCurrManager'], label='Employee Who Stayed', shade=True, color='b')
        plt.legend()
        st.pyplot()
    if not st.checkbox("Hide KDE (Kernel Density Estimate) - Total Working Years", True, key='twy'):
        plt.figure(figsize=(12,7))
        sns.kdeplot(df_left['TotalWorkingYears'], label='Employee Who Left', shade=True, color='r')
        sns.kdeplot(df_stayed['TotalWorkingYears'], label='Employee Who Stayed', shade=True, color='b')
        plt.legend()
        st.pyplot()

elif opt =="Predictive Analysis":
    st.header("Predictive Analysis using Machine Learning.")
    if not st.checkbox("Hide Logistic Regression", True, key='lreg'):
        st.write("This analysis is performed using Logistic Regression. Please adjust the below hyper parameters to fine tune the algorithm.")
        C = st.number_input('C (Regularization parameter (0.01 to 10))', min_value=0.01, max_value=10.0, step=0.01, key='C_LR')
        max_iter = st.slider("Maximum number of iteration (100 to 500)", 100, 500, key="max_iter")
        if st.button("Classify", key="classify"):
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy of predicition is {:.2f} %".format(100*accuracy_score(y_pred,y_test)))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True)
            st.pyplot()
    if not st.checkbox("Hide Support Machine Classification (SVC)", True, key='svc'):
        st.write("This analysis is performed using Support Machine Classification (SVC). Please adjust the below hyper parameters to fine tune the algorithm.")
        C = st.number_input('C (Regularization parameter (0.01 to 10))', min_value=0.01, max_value=10.0, step=0.01, key='C')
        kernel = st.radio("Kernel",("linear","rbf"),key="kernel")
        gamma = st.radio("Gamma (Kernel Coefficient)",("scale","auto"), key="gamma")
        if st.button("Classify", key="classify"):
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy of predicition is {:.2f} %".format(100*accuracy_score(y_pred,y_test)))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True)
            st.pyplot()    

st.sidebar.write("---")
st.sidebar.write("Please share your thoughts with me: leesaymoon@gmail.com")
