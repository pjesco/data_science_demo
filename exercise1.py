"""
In-Class Activity: Exploratory Data Analysis (EDA) with Starbucks Drinks

Dataset: starbucks_drinkMenu_expanded.csv (available in Canvas)
Goal: Understand what the dataset contains, identify patterns, and create a meaningful visualization.
"""

print("*************** Part 1 ***************")

"""
1. Load the CSV file into a pandas DataFrame
2. Print:
- The full DataFrame (⚠️ observe what happens)
- The first 5 rows
- The last 5 rows

3. Display:
- Column names
- Dataset info (data types + missing values)
- Descriptive statistics

💡 Reflection questions:

A) Why is printing the entire DataFrame usually a bad idea?
B) Which columns are numeric? Which are categorical?
C) Do the statistics make sense for food and drinks?
"""

import pandas as pd

df = pd.read_csv("logs/starbucks_drinkMenu_expanded.csv")
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
print(df)
print(df.head())
print(df.tail())

print(df.columns)
print(df.info())
print(df.describe())

print("*************** Part 2 ***************")
"""
1. Select and print only these columns:
- Beverage
- Caffeine (mg)
- Sodium (mg)

Display:
- One specific row (your choice)
- Two different rows (your choice)
- Those same rows, but only for two columns

💡 Reflection questions:
A) When would you select rows vs columns?
B) What kind of question would each selection help answer?
"""
print(df[['Beverage','Caffeine (mg)','Sodium (mg)']])
#print(df.iloc[[100]])
#print(df.iloc[[101,102]])
print(df.loc[10])
print(df.loc[[10,20]])
print(df[['Beverage','Sodium (mg)']].iloc[[101,102]])

print("*************** Part 3 ***************")

"""
1. Create a table showing where missing values exist
2. Count how many missing values each column has

💡 Reflection questions:

A) Which columns have missing values?
B) Would you drop, fill, or ignore them? Why?
"""

missing_values = {
    "Feature": df.columns,
    "Missing Values": df.isna().sum(),
    "Percentage of Missing":(df.isna().sum()/len(df)*100)
}
print(pd.DataFrame.from_dict(missing_values, orient="index"))
#print(df.isnull())
#print(df.isnull().sum())

print("*************** Part 4 ***************")
"""
1. Filter the dataset to show beverages with:
- Calories below 100
- Improve the filter to show beverages with:
- Calories below 100 AND
- Caffeine below 50 mg

⚠️ Challenge:
Make sure your condition works correctly. If it doesn’t, debug it.

💡 Reflection questions:
A) What kind of customer might this filter represent?
B) Why do we need parentheses in compound conditions?
"""
print(df[df["Calories"] < 100])
condition1 = df["Calories"] < 100
condition2 = df["Caffeine (mg)"] < 50
print(df[condition1 & condition2])
print(df[condition1 & ~condition2])

print("*************** Part 5 ***************")
"""
1. Group the dataset by Beverage_category
2. Compute the average calories per category
3. Print the result

💡 Reflection questions:
A) Which category is the most calorie-dense on average?
B) Why is the mean a reasonable (or not) choice here?
"""

print(df.groupby("Beverage_category")['Calories'].mean())

print("*************** Part 6 ***************")

"""
1. Create a new DataFrame containing only numeric nutrition columns
2. Compute the correlation matrix
3. Plot a correlation heatmap using Plotly

💡 Reflection questions:

A) Which nutrients are strongly correlated?
B) Are there any surprising relationships?
C) Why does correlation not imply causation?
"""

n_df = df.select_dtypes(include='number')
print(n_df.head())

corr = n_df.corr()

import plotly.express as px

fig = px.imshow(
    corr,
    text_auto=True,
    title='Correlation Matrix Heatmap'
)
#fig.show()

fig2 = px.box(
    df,
    x="Caffeine (mg)"
)
#fig2.show()
print(df.info())
#Group by beverage category and display average caffeine value per category
print(df.groupby('Beverage_category')['Caffeine (mg)'].mean())

missing_data = df[df['Caffeine (mg)'].isnull()]
print(missing_data)

# Strategy 1
df_dropped = df.dropna(subset=['Caffeine (mg)'])
print(f'Original dataset shape: {df.shape}. Dropped dataset shape: {df_dropped.shape}.')

# Strategy 2
df_filled = df.copy()
mean_caffeine = df['Caffeine (mg)'].mean()
df_filled['Caffeine (mg)'].fillna(mean_caffeine)
print(f'Original dataset shape: {df.shape}. Filled dataset shape: {df_filled.shape}.')

#Stretagy 3
df_zero = df.copy()
df_zero['Caffeine (mg)'].fillna(0)
print(f'Original dataset shape: {df.shape}. Zero dataset shape: {df_zero.shape}.')

# Strategy 4
df_category = df.copy()
df_category['Caffeine (mg)'] = df_category.groupby(
    'Beverage_category')['Caffeine (mg)'].transform(
        lambda x: x.fillna(x.mean())
    )
print(df_category['Caffeine (mg)'].describe())
