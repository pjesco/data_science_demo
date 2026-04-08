import math
from scipy.stats import norm
import numpy as np

# Example 1
mu1 = 29
n = 50
sigma1 = 4

z_score1 = (28.3-mu1)/(sigma1/math.sqrt(n))
print(z_score1)
p_value1 = norm.cdf(z_score1)
print(p_value1)

if (p_value1 < 0.05):
    print("Reject the null hypothesis")
else:
    print('Fail to reject the null hypothesis')

# Exercise 2
n = 50
mu2 = 30
sigma2 = 4

z_score2 = (28.3-mu2)/(sigma2/math.sqrt(n))
print(z_score2)
p_value2 = 2*norm.cdf(z_score2) # 2 tailed
print(p_value2)

if (p_value2 < 0.05):
    print("Evidence against the null hypothesis")
else:
    print('Fail to reject the null hypothesis')

# Example 3
n1, mean1, std1 = 40, 8.4, 4.2
n2, mean2, std2 = 40, 6.9, 3.8

se = np.sqrt(std1**2/n1 + std2**2/n2)
z_score3 = (mean1-mean2)/se
print(z_score3)
p_value3 = 1 - norm.cdf(z_score3)
print(p_value3)
if (p_value3 < 0.05):
    print("Evidence against the null hypothesis")
else:
    print('Fail to reject the null hypothesis')

# Example 4
from scipy import stats
one_sample_data = [177.3, 182.7, 169.6, 176.3, 180.3, 179.4, 178.5, 177.2, 181.8, 176.5]

result4 = stats.ttest_1samp(one_sample_data, 175.3, alternative='two-sided')
# The output of this function is tuple containing
# the t-statistic and the p-value
print(f't-statistic of Example 4: {result4.statistic:.3f}')
print(f'p-value of Example 4: {result4.pvalue:.3f}')

if result4.pvalue < 0.05:
    print('Evidence to reject null hypothesis')
else:
    print('Fail to reject null hypothesis.'
          'No evidence against it.')
    
# Example 5
import plotly.express as px
import pandas as pd

df = px.data.tips()
print(df.head())
print(df.groupby('smoker')['tip'].mean())

# Take advantage of this dataset and practive plotly for data visualization
# for example, histogram, scatter, boxplots, etc.

smokers = df[df['smoker'] == 'Yes']['tip']
non_smokers = df[df['smoker'] == 'No']['tip']

result5 = stats.ttest_ind(smokers, non_smokers, equal_var=False)

if result5.pvalue < 0.05:
    print('Evidence to reject null hypothesis')
else:
    print('Fail to reject null hypothesis.'
          'No evidence against it.')
    
# Example 6
n6 = 15

before = np.random.normal(loc=150,scale=10,size=n6)
after = before - np.random.normal(loc=5,scale=5,size=n6)

difference = after - before

print(f'Mean difference = {difference.mean()} (after - before)')

result6 = stats.ttest_rel(after, before)

if result6.pvalue < 0.05:
    print('Evidence to reject null hypothesis')
else:
    print('Fail to reject null hypothesis.'
          'No evidence against it.')
