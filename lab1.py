import pandas as pd
import plotly.express as px


# Step 1
df = pd.read_csv('logs/oct25-2024.csv')

print(df.head())

print(df.info())
print(df.shape)
print(df.dtypes)

# Step 2
print(df.describe())

# TSS mg/L seems to be constant at 0
# TDS mg/L and the Cond categories have very high std devs and vary wildly
# Depth seems to have both positive and negative values, which, depending on the
# refence point, I would think should be only one sign

# Step 3
cov_mat = df.select_dtypes(include='number').cov()
fig_cov = px.imshow(
    cov_mat,
    text_auto=True,
    title='Covariance Matrix'
)
fig_cov.show()

corr_mat = df.select_dtypes(include='number').corr()
fig_corr = px.imshow(
    corr_mat,
    text_auto=True,
    title='Correlation Matrix'
)
fig_corr.show()

# Some with the highest positive corr are Cond uS/cm x nLF Cond uS/cm,
# Depth x Pressure, Cond uS/cm x Sal psu, Cond uS/cm x SpCond uS/cm,
# Cond uS/cm x TDS mg/L, Sal psu x TDS mg/L

# Some with the highest negative corr are all of the ODO's with Pressure,
# Sal psu, and SpCond uS/cm; pH x pH mV; Longitude x Temp C; Longitude x pH mV

# Step 4
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

num_features = df.select_dtypes(include='number')
outliers_dictionary = {}
for f in num_features:
    outliers, lb, ub = detect_outliers(df, f)
    outliers_dictionary[f] = {
        "lower bound": lb,
        "upper bound": ub,
        "number of outliers": len(outliers),
        "percentage of outliers": len(outliers)/len(df)*100,
    }

outliers_df = pd.DataFrame.from_dict(outliers_dictionary, orient="index")
print(outliers_df)
print(f'Total outliers: {outliers_df['number of outliers'].sum()}')

# Found a total of 1068 outliers
# Latitude, ODO % sat, and ODO % CB had the highest percentage with around 27%,
# I think categories with high percentages like these should not be corrected
# or removed because the represent too much of the total data
# I think ones with a much smaller share of the data, maybe around 3% can be removed

df_clean = df.copy()
clean_features = []
for f in num_features:
    if outliers_dictionary[f]['percentage of outliers'] < 3:
        clean_features.append(f)

def clean_outliers(df_clean, features):
    for f in features:
        Q1 = df[f].quantile(0.25)
        Q3 = df[f].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        condition1 = df_clean[f] >= lower_bound
        condition2 = df_clean[f] <= upper_bound
        df_clean = df_clean[condition1 & condition2]
    
    return df_clean

df_clean = clean_outliers(df_clean, clean_features)
print(df_clean)

# Step 5
fig_scatter = px.scatter(
    df_clean,
    x = 'Sal psu',
    y = 'Temp °C',
    title='Salinity psu vs Temperature °C Scatter'
)
fig_scatter.show()

fig_hist = px.histogram(
    df_clean,
    x = 'pH',
    title='pH Histogram'
)
fig_hist.show()

# In the histogram, I notice that pH has a by far the largest readings between 
# 8.4 and 8.45, with a small distribution outside of that range
# In the scatter plot, there does not seem to be a strong correlation in the data

# Step 6
import streamlit as st
st.title('Lab 1')
st.header('Parker Esco')

show_data = st.checkbox('Show Data')
if show_data:
    st.dataframe(df)

st.subheader("Summary Statistics")
st.dataframe(df.describe())
st.plotly_chart(fig_corr)
with st.expander("Scatter Plot"):
    st.plotly_chart(fig_scatter)