import pandas as pd

df = pd.read_csv("logs/biscayne_bay_water_quality.csv")

print(df.head())
print(df.tail())

print(df.columns)
print(df.shape)

print(df.dtypes)

print(df.isna().sum())

print(df["Temp C"])
print("Min: ",df["Temp C"].min())
print(df["Temp C"].max())
print(df["Temp C"].mean())
print(df["Temp C"].std())

print(df.describe())

#Filtering

print(df[df["Temp C"] > 24.5])

print(df[["Latitude","Longitude","TWC"]])

print(df.describe().T[["min","max","mean"]])

# Do you see potential outliers?
# Check minimum and maximum values of all parameeters? Anything unusual?
# What percentage of my data is questionable?
## Domain rule: values of salinity should be above 30

invalid_salinity = df[df["Salinity"] < 30]
print(len(invalid_salinity)/len(df)*100)

clean_df = df[df["Salinity"].between(30,45)]
print(clean_df.shape)

print(clean_df.describe())
print(clean_df.corr(numeric_only=True))