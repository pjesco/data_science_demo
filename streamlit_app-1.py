import streamlit as st
import nltk
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os
import plotly.express as px


load_dotenv()

api_key = os.getenv("NEWS_API")

for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "vader_lexicon"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

POS_EXPLANATIONS = {
            "CC": "Coordinating conjunction",
            "CD": "Cardinal number",
            "DT": "Determiner",
            "EX": "Existential there",
            "FW": "Foreign word",
            "IN": "Preposition/subordinating conjunction",
            "JJ": "Adjective",
            "JJR": "Adjective, comparative",
            "JJS": "Adjective, superlative",
            "LS": "List item marker",
            "MD": "Modal",
            "NN": "Noun, singular or mass",
            "NNS": "Noun, plural",
            "NNP": "Proper noun, singular",
            "NNPS": "Proper noun, plural",
            "PDT": "Predeterminer",
            "POS": "Possessive ending",
            "PRP": "Personal pronoun",
            "PRP$": "Possessive pronoun",
            "RB": "Adverb",
            "RBR": "Adverb, comparative",
            "RBS": "Adverb, superlative",
            "RP": "Particle",
            "SYM": "Symbol",
            "TO": "to",
            "UH": "Interjection",
            "VB": "Verb, base form",
            "VBD": "Verb, past tense",
            "VBG": "Verb, gerund/present participle",
            "VBN": "Verb, past participle",
            "VBP": "Verb, non-3rd person singular present",
            "VBZ": "Verb, 3rd person singular present",
            "WDT": "Wh-determiner",
            "WP": "Wh-pronoun",
            "WP$": "Possessive wh-pronoun",
            "WRB": "Wh-adverb",
        }


def tag_color(tag):
    if tag.startswith("NN"):  return "#f0c040"  # nouns — gold
    if tag.startswith("VB"):  return "#5ba4f5"  # verbs — blue
    if tag.startswith("JJ"):  return "#56c997"  # adjectives — green
    if tag.startswith("RB"):  return "#d98cf0"  # adverbs — purple
    if tag in ("IN", "CC"):   return "#f07850"  # conjunctions/prepositions — orange
    if tag.startswith("PRP"): return "#e87d7d"  # pronouns — red
    if tag == "DT":           return "#7eb8d4"  # determiners — light blue
    return "#888888"  # everything else — grey

def render_tokens(tokens, bg="#1e3a5f", text_color="white"):
    html = "<div style='margin-bottom:1rem;'>"
    for t in tokens:
        html += (
            f"<span style='display:inline-block;"
            f"background:{bg};"
            f"color:{text_color};"
            f"padding:6px 10px;"
            f"margin:4px;"
            f"border-radius:8px;"
            f"font-size:0.9rem;'>"
            f"{t}"
            f"</span>"
        )
    html += "</div>"
    return html

st.set_page_config(
    page_title="Introduction to NLP",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }


/* Optional: make header transparent/minimal instead of removing it */
[data-testid="stHeader"] {
    background: transparent;
    height: 0.5rem;
}

/* Main background */
.stApp { 
background: #0f1d35; color: #e8dfc8; 
}


/* Main content area */
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a1628;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #e8dfc8 !important; }

/* Radio buttons in sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.3rem 0;
    font-size: 0.92rem;
}



.pos-box {
    background: #f7f3e9;
    border: 1px solid #d8ccb2;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.pos-pill {
    display: inline-block;
    margin: 0.35rem 0.35rem 0.35rem 0;
    padding: 0.45rem 0.7rem;
    border-radius: 10px;
    color: #1c1c1c;
    font-size: 0.92rem;
    font-weight: 500;
    line-height: 1.4;
    border: 1px solid rgba(0,0,0,0.08);
}

/* Headings */
h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #f0c040;
}
h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #e8dfc8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
}
h3 { font-size: 1.05rem; color: #c8bfa8; }
p, li { color: #c8bfa8; }

/* Module tag */
.tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #f0c040;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* Buttons */
.stButton > button {
    background: #1e3a5f;
    color: #f0c040;
    border: 1px solid #f0c040;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.stButton > button:hover {
    background: #f0c040;
    color: #0a1628;
}

/* Text inputs */
.stTextArea textarea, .stTextInput input {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8dfc8 !important;
    font-family: 'IBM Plex Mono', monospace;
}

hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    #st.logo("fiu-logo.png")
    st.markdown("""
    <div style='padding:1rem 0 0.8rem 0;'>
        <div style='font-family:"Playfair Display",serif; font-size:1.25rem;
                    color:#f0c040; font-weight:700; line-height:1.4;'>
            Intro to NLP<br>
        </div>
        <div class='tag'><br>FIU - Data Science <br> Prof. Gregory Murad Reis</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Text Preprocessing & Analysis",
        "Frequency Distributions",
        "Wordcloud",
        "Sentiment Analysis",
        "News API",
    ], label_visibility="collapsed")

if page == "Text Preprocessing & Analysis":
    st.markdown('<div class="tag">Module 00</div>', unsafe_allow_html=True)
    st.title("How We Break Down Text")
    default = """I am the best data science student at Florida International University!"""
    text = st.text_area("Input paragraph",
                        value=default,
                        height=110,
                        label_visibility="collapsed")

    if st.button("Analyze →"):
        tokens = word_tokenize(text)
        non_stop_tokens = []
        for token in tokens:
            if token.lower() not in stopwords.words("english") and token.isalpha():
                non_stop_tokens.append(token)

        tagged_tokens = pos_tag(non_stop_tokens)

        st.subheader("Step 1 — All Words")
        st.markdown(render_tokens(tokens), unsafe_allow_html=True)

        st.subheader("Step 2 — After Removing Stopwords")
        st.markdown(render_tokens(non_stop_tokens, bg="#f0c040", text_color="#1c1c1c"), unsafe_allow_html=True)

        st.subheader("Step 3 — Part-of-Speech Tags")

        html = "<div class='pos-box'>"

        for word, tag in tagged_tokens:
            color = tag_color(tag)
            meaning = POS_EXPLANATIONS.get(tag, "Other")
            html += (
                f"<span class='pos-pill' style='background:{color};'>"
                f"<b>{word}</b> ({tag}) - {meaning}"
                f"</span>"
            )

        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

elif page == "Frequency Distributions":
    st.markdown('<div class="tag">Module 01</div>', unsafe_allow_html=True)
    st.title("Frequency Distributions")

    selected_file = st.selectbox(
        "Choose a book",
        gutenberg.fileids()
    )
    if st.button("Get Frequency Distribution →"):
        with st.spinner("Loading..."):
            text = gutenberg.raw(selected_file)
            tokens = word_tokenize(text)

            clean_tokens = [
                t.lower()
                for t in tokens
                if t.isalpha() and t.lower() not in stopwords.words("english")
            ]

            freq = FreqDist(clean_tokens)

            top_words = freq.most_common(10)
            df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

            st.bar_chart(df.set_index("Word"))

elif page == "Wordcloud":
    st.markdown('<div class="tag">Module 02</div>', unsafe_allow_html=True)
    st.title("Wordcloud")

    selected_file = st.selectbox(
        "Choose a book",
        gutenberg.fileids()
    )
    if st.button("Generate Wordcloud →"):
        with st.spinner("Loading..."):
            text = gutenberg.raw(selected_file)
            tokens = word_tokenize(text)

            clean_tokens = [
                t.lower()
                for t in tokens
                if t.isalpha() and t.lower() not in stopwords.words("english")
            ]
            clean_text = " ".join(clean_tokens)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(clean_text)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

elif page == "Sentiment Analysis":
    st.markdown('<div class="tag">Module 05</div>', unsafe_allow_html=True)
    st.title("Sentiment Analysis")

    text = st.text_area(
        "Enter a sentence",
        value="I love FIU so much!",
        height=100
    )

    if st.button("Analyze Sentiment →"):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)

        st.subheader("Sentiment Scores")
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "😊 Positive"
        elif compound <= -0.05:
            sentiment = "😠 Negative"
        else:
            sentiment = "😐 Neutral"

        st.write(sentiment)
        st.markdown("### Sentiment Breakdown")

        st.markdown(
            f"""
            <div style='display:flex; gap:10px;'>
                <div style='background:#56c997; padding:10px; border-radius:8px;'>Positive: {scores['pos']}</div>
                <div style='background:#c8bfa8; padding:10px; border-radius:8px;'>Neutral: {scores['neu']}</div>
                <div style='background:#e87d7d; padding:10px; border-radius:8px;'>Negative: {scores['neg']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "News API":
    st.markdown('<div class="tag">Playground</div>', unsafe_allow_html=True)
    st.title("News API")

    tab1, tab2 = st.tabs(["Top Stories", "Most Popular Articles"])
    with tab1:
        st.subheader("Top Stories API")
        topics = ['Technology','Science','Buisiness','Health','Sports','Entertainment','Politics']
        topic = st.selectbox('Choose a topic',
                     topics)
        keyword = st.text_input('Input keyword')
        tk = topic + '-' + keyword

        api = st.selectbox('Select API',
                           ['NewsAPI'])
        url = f'https://newsapi.org/v2/everything?q={tk}&apiKey={api_key}'

        if st.button('Fetch Articles'):
            response = requests.get(url).json()

            if response['status'] == 'error':
                st.error(response['message'])
            elif len(response['articles']) == 0:
                st.error('Search returned no articles')
            else:
                articles = {}
                for a in response['articles']:
                    articles[a['title']] = a['content'].split()

                combined = []
                for t in response['articles']:
                    temp2 = ' '.join(articles[t['title']])
                    temp1 = t['description']
                    combine = t['title']
                    if type(temp1) == str:
                        combine += ' ' + temp1
                    combine += ' ' + temp2
                    combined.append(combine)
                combined = ' '.join(combined)

                words = word_tokenize(combined)

                filtered_text = []
                for i in words:
                    if i.lower() not in stopwords.words('english') and i.isalpha():
                        filtered_text.append(i)

                filtered_message = ' '.join(filtered_text)

                frequency_dist = nltk.FreqDist(filtered_text)
                #print(frequency_dist.most_common(10))
                df = pd.DataFrame(frequency_dist.most_common(10), columns=['Word','Frequency'])

                st.dataframe(df)

                fig = px.bar(df,
                            x='Word',
                            y='Frequency',
                            title='Top 10 Most Frequent Words')
                st.plotly_chart(fig)

                st.subheader('Wordcloud')
                wc = WordCloud(width=1200,
                                height=800,
                                background_color='white'
                                ).generate(filtered_message)
                fig1, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig1)
                
                st.subheader('Articles')
                for a in response['articles']:
                    st.write(a['title'])
                    st.write(a['source']['name'])
                    if type(a['description']) == str:
                        st.write(a['description'])
                    if type(a['url']) == str:
                        st.write(a['url'])
                    st.divider()
        st.subheader('Reflection')
        st.markdown('For testing I searched for articles about technology with NewsAPI.')
        st.markdown('AI, li, chars, technology, and Nvidia and Anthropic appeared most often.')
        st.markdown('These worrds make sense, especially the frequency of AI and their associated companies.')
        st.markdown('The API results worked as expected and returned articles about technology.')
            


    with tab2:
        st.subheader("Most Popular Articles API")


