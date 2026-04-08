import pandas as pd
import numpy as np
import plotly.express as px

pd.set_option('display.max_columns', None) #Visualize all columns in terminal

df = pd.read_csv("logs/biscayne_bay_water_quality2.csv")

print('Features: \n', df.columns)
print('Top 5 rows: \n', df.head())
print('Sum of Null values: \n', df.isnull().sum())
print('Sum of duplicate Values: \n', df.duplicated().sum())
print('Descriptive Statistics: \n', df.describe())
print(f'Mean Temperature: \n  {df["Temperature (c)"].mean():.2f}')
print('Median Temperature: \n',df["Temperature (c)"].median())
print('Mode Temperature: \n', df["Temperature (c)"].mode())
print('Variance Temperature: \n', df["Temperature (c)"].var())
print('Standard Deviation Temperature: \n', df["Temperature (c)"].std())

some_features = ["Total Water Column (m)",
                 "Temperature (c)",
                 "Salinity (ppt)",
                 "pH",
                 "ODO mg/L"]

for f in some_features:
    fig = px.box(df,
                y=f,
                title=f"Box Plot of {f}",
                labels={f:f})
    #fig.show()

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
for f in some_features:
    outliers, lb, ub = detect_outliers(df, f)
    outliers_dictionary[f] = {
        "lower bound": lb,
        "upper bound": ub,
        "number of outliers": len(outliers),
        "percentage of outliers": len(outliers)/len(df)*100,
    }

outliers_df = pd.DataFrame.from_dict(outliers_dictionary, orient="index")
print(outliers_df)

#Remove the outliers
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

print(df_clean.describe())

for f in some_features:
    fig2 = px.box(
        df_clean,
        x=f,
        title=f"Box Plot of {f}"
    )
    #fig2.show()

# Covariance between Temperature and Salinity
cov_matrix = df_clean[["Temperature (c)","Salinity (ppt)"]].cov()
cov_temp_sal = cov_matrix.loc["Temperature (c)","Salinity (ppt)"]