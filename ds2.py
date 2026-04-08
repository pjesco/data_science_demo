import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""Example 1"""
data = {
    'year': [2010, 2011, 2012, 2010, 2011, 2012, 2010, 2011, 2012],
    'team': ['FCBarcelona', 'FCBarcelona', 'FCBarelona', 'RMadrid', 'RMadrid', 'RMadrid', 'ValenciaCF', 'ValenciaCF', 'ValenciaCF'],
    'wins': [30,28,32,29,32,26,21,17,19],
    'draws': [6,7,4,5,4,7,8,10,8],
    'losses': [2,3,2,4,2,5,9,11,11]
}
#print(data)
df = pd.DataFrame(data)
#print(df)

"""Example 2"""
edu = pd.read_csv('logs/educ_figdp_1_Data.csv',
                  usecols=['TIME','GEO','Value'],
                  na_values=":")
print(edu)
print(edu.head())
print(edu.tail())

print(edu['Value'])
print(edu[10:14])
print(edu.loc[90:94, ['TIME','GEO']])

print(edu[edu['Value'] > 6.5])

print(edu[edu['Value'].isnull()])

print(edu.max(axis=0)) #function should be applied to the rows for each column

s = edu['Value']/100
print(s)

s = edu['Value'].apply(np.sqrt)
print(s.head())

s = edu['Value'].apply(lambda d: d**2)
print(s.head())

edu['ValueNorm'] = edu['Value']/edu['Value'].max()

eduAlt = edu.drop('ValueNorm', axis=1)

edu = pd.concat([edu, 
                 pd.DataFrame({'TIME':2000, 'Value':5.00, 'GEO':'a'},
                 index=[max(edu.index)+1])])

some_data = {
    'A': [1, None, 3, None],
    'B': [None, None, 6, None],
    'C': [7,8,9,None]
}

some_df = pd.DataFrame(some_data)
print(some_df)

df_any = some_df.dropna(how='any')
print(df_any)

df_all = some_df.dropna(how='all')
print(df_all)

edu_drop = edu.dropna(how='any', subset=['Value'])
print(edu_drop.head())

eduFilled = edu.fillna(value={'Value':0})
print(eduFilled.head())

edu.sort_values(by='Value', ascending=False, inplace=True)
print(edu.head())

edu.sort_index(axis=0,ascending=True,inplace=True)
print(edu.head())

group = edu[['GEO','Value']].groupby('GEO').mean()
print(group)

filtered_edu = edu[edu['TIME'] > 2005]

pivedu = pd.pivot_table(filtered_edu,
                         values='Value',
                         index=['GEO'],
                         columns=['TIME']
)
print(pivedu)

print(pivedu.loc[['Spain','Portugal'],[2006,2011]])

pivedu = pivedu.drop([
    "Euro area (13 countries)",
    "Euro area (15 countries)",
    "Euro area (17 countries)",
    "Euro area (18 countries)",
    "European Union (25 countries)",
    "European Union (27 countries)",
    "European Union (28 countries)",
], axis=0)

print(pivedu.head())

pivedu = pivedu.rename(index={"Germany (until 1990 former territory of the FRG)":'Germany'})
print(pivedu.head(15))

print(pivedu.rank(ascending=False, method='first'))

sample_data = pd.Series([50,30,20,50,40])
print(sample_data.rank(method='average'))