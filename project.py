import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords

# Install required packages (Run only once)
# pip install streamlit pandas matplotlib wordcloud scikit-learn seaborn vaderSentiment transformers emoji nltk

# Set up Streamlit
st.title("Twitter Sentiment Analysis and Visualization")
st.sidebar.header("Settings")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Loaded Successfully", data.head())

    # Clean the text
    def clean_text_without_tokenization(text):
        text = emoji.demojize(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@[\w]+', '', text)  # Remove mentions
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        stop_words_id = set(stopwords.words('indonesian'))
        text = ' '.join([word for word in text.split() if word not in stop_words_id])  # Remove stopwords
        return text

    data['Cleaned_Tweet_Content'] = data['Tweet_Content'].apply(clean_text_without_tokenization)

    st.subheader("Cleaned Tweet Data")
    st.write(data[['Tweet_Content', 'Cleaned_Tweet_Content']].head())

    # WordCloud Visualization
    all_tweets = " ".join(data['Cleaned_Tweet_Content'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets)

    st.subheader("WordCloud of Tweets")
    st.image(wordcloud.to_image())

    # Sentiment Analysis using VADER
    analyzer = SentimentIntensityAnalyzer()

    def vader_sentiment(tweet):
        vader_score = analyzer.polarity_scores(tweet)['compound']
        if vader_score > 0.005:
            return 'positive'
        elif vader_score < -0.005:
            return 'negative'
        return 'neutral'

    data['Sentiment'] = data['Cleaned_Tweet_Content'].apply(vader_sentiment)
    st.subheader("Sentiment Distribution")
    sentiment_counts = data['Sentiment'].value_counts()
    st.write(sentiment_counts)

    # Sentiment Distribution Plot
    sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'], title='Sentiment Distribution')
    st.pyplot()

    # Model Sentiment Analysis (using IndoBERTweet)
    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def classify_sentiment(text):
        if pd.isna(text):
            return None
        result = sentiment_analyzer(str(text))[0]
        return result['label']

    data['Model_Sentiment'] = data['Cleaned_Tweet_Content'].apply(classify_sentiment)

    st.subheader("Model Sentiment")
    st.write(data[['Cleaned_Tweet_Content', 'Sentiment', 'Model_Sentiment']].head())

    # Visualizing Sentiment Analysis Results
    sentiment_comparison = pd.crosstab(data['Sentiment'], data['Model_Sentiment'])
    st.subheader("Sentiment Comparison between VADER and Model")
    st.write(sentiment_comparison)

    # Topic Modeling using BERTopic
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')
    topic_model = BERTopic(language="indonesia", calculate_probabilities=True, embedding_model=embedding_model)

    # Preprocessing text data for BERTopic
    texts = data['Cleaned_Tweet_Content'].dropna().astype(str).tolist()
    topics, _ = topic_model.fit_transform(texts)

    data['Topic'] = topics
    st.subheader("Topic Distribution")
    topic_counts = data['Topic'].value_counts()
    st.write(topic_counts)

    # Topic Visualization
    st.subheader("Topic Visualization")
    fig = topic_model.visualize_topics()
    st.pyplot(fig)

    # Exporting Data to CSV
    export_file = st.sidebar.button("Export Cleaned Data")
    if export_file:
        data.to_csv("cleaned_data_with_sentiment_and_topics.csv", index=False)
        st.success("Data exported successfully!")

# Add a footer
st.sidebar.markdown("Made with ❤️ by Streamlit")
