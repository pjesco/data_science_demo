import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Heart Study Data Science Project')
st.header('Parker Esco and Andres Acosta')

st.divider()

st.sidebar.header("Load Datasets")
file_upload = st.sidebar.file_uploader("Upload a file", type=["csv"])
if file_upload is not None:
    df = pd.read_csv(file_upload)
else:
    df = pd.read_csv("logs/framingham.csv")

tab1, tab2, tab3 = st.tabs(["Raw Data","EDA","Hypothesis Testing"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df)
    st.caption("Framingham Heart Study Data")

    st.divider()

    st.markdown(f'Data Shape: {df.shape}')
    st.markdown(f'Data Columns: {df.columns}')

    st.markdown(f'Info:')
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(info_df)

    st.divider()

    for f in df.columns:
        df[f].fillna(df[f].mean(), inplace=True)
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(info_df)
    st.caption('Null values filled using column mean')

    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        condition1 = df[column] < lower_bound
        condition2 = df[column] > upper_bound
        outliers = df[condition1 | condition2]
        return outliers, lower_bound, upper_bound
    
    outliers_dictionary = {}
    some_features = ['education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
    for f in some_features:
        outliers, lb, ub = detect_outliers(df, f)
        outliers_dictionary[f] = {
            "lower bound": lb,
            "upper bound": ub,
            "number of outliers": len(outliers),
            "percentage of outliers": len(outliers)/len(df)*100,
        }
    outliers_df = pd.DataFrame.from_dict(outliers_dictionary, orient="index")
    st.dataframe(outliers_df)

    df_clean = df.copy()

    for f in some_features:
        Q1 = df[f].quantile(0.25)
        Q3 = df[f].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        condition1 = df_clean[f] >= lower_bound
        condition2 = df_clean[f] <= upper_bound
        df_clean = df_clean[condition1 & condition2]

with tab2:
    st.subheader('Exploratory Data Analysis and Data Visualization')

    st.dataframe(df_clean.describe())
    st.caption('Summary Statistics')

    st.divider()

    fig_hist1 = px.histogram(df_clean,
                             x='education',
                             title='Participant Education Distribution')
    st.plotly_chart(fig_hist1)

    fig_hist2 = px.histogram(df_clean,
                             x='age',
                             title='Age Distribution')
    st.plotly_chart(fig_hist2)

    fig_hist3 = px.histogram(df_clean,
                             x='cigsPerDay',
                             title='Cigarettes Per Day Distribution')
    st.plotly_chart(fig_hist3)

    corr = df_clean.corr(numeric_only=True)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm',fmt='.2f',annot_kws={'size':6})

    st.subheader('Correlation Matrix Heatmap')
    st.pyplot(fig)

    fig_scatter1 = px.scatter(df_clean,
                        x='sysBP',
                        y='diaBP',
                        title='SysBP vs. DiaBP')
    st.plotly_chart(fig_scatter1)

    fig_scatter2 = px.scatter(df_clean,
                              x='age',
                              y='cigsPerDay',
                              title='Age vs. Cigarettes Per Day')
    st.plotly_chart(fig_scatter2)

    fig_box1 = px.box(df_clean,
                      y=['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose'],
                      title='Box Plot')
    st.plotly_chart(fig_box1)

    fig_scatter2 = px.scatter(df_clean,
                                 x='totChol',
                                 y='heartRate',
                                 title='Scatter of totChol vs. Heart Rate')
    st.plotly_chart(fig_scatter2)

    fig_scatter3 = px.scatter(df_clean,
                                 x='BMI',
                                 y='heartRate',
                                 title='Scatter of BMI vs. Heart Rate')
    st.plotly_chart(fig_scatter3)

    fig_bar1 = px.bar(df_clean,
                      x='male',
                      y='currentSmoker',
                      title='Bar Chart of Sex vs. Smoker')
    st.plotly_chart(fig_bar1)


