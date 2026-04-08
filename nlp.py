import nltk
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

text = """ Data science is a multidisciplinary field that extracts actionable insights from structured and unstructured data using statistics, machine learning, and data analysis. It involves the entire data lifecycle—cleaning, mining, and modeling—to inform decision-making, with high demand for skills in Python, SQL, and data visualization. It's a rapidly growing career with a median salary of $112,590, expected to grow 34% from 2024 to 2034. """

#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

words = word_tokenize(text)
print(words)

# We are going to remove stopwords and punctuation marks

filtered_text = []
for i in words:
    if i.lower() not in stopwords.words('english') and i.isalpha():
        filtered_text.append(i)

print(filtered_text)
filtered_message = ' '.join(filtered_text)

from wordcloud import WordCloud
fig1 = WordCloud(width=1200,
                 height=800,
                 background_color='white'
                 ).generate(filtered_message)
#plt.figure(figsize=(10,7))
#plt.imshow(fig1)
#plt.axis('off')
#plt.show()

frequency_dist = nltk.FreqDist(filtered_text)
print(frequency_dist.most_common(10))

import pandas as pd
import plotly.express as px

df = pd.DataFrame(frequency_dist.most_common(10), columns=['word','count'])
fig = px.bar(df,
             x='word',
             y='count',
             title='Top 10 Most Frequent Words')
#fig.show()

#nltk.download('gutenberg')
#print(gutenberg.fileids())

ebook = gutenberg.raw('shakespeare-hamlet.txt')

ebook_tokens = word_tokenize(ebook)

filtered_ebook = []
for i in ebook_tokens:
    if i.isalpha() and i.lower() not in stopwords.words('english'):
        filtered_ebook.append(i)

filtered_ebookString = ' '.join(filtered_ebook)
fig3 = WordCloud(width=1200,
                 height=800,
                 background_color='white'
                 ).generate(filtered_ebookString)
#plt.figure(figsize=(10,7))
#plt.imshow(fig3)
#plt.axis('off')
#plt.show()

# TODO: Generate a bar chart for most frequently used words
# don't forget to filter out MORE stopwords seen in wordcloud
# frequency_dist_ebook = nltk.FreqDist(filtered_ebook)

from nltk.sentiment import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

scores = sia.polarity_scores('See the child')
print(scores)