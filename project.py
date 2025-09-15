import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --------------------
# App header
# --------------------
st.title("Twitter Sentiment and Topic Modeling")
st.sidebar.header("Settings")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your labeled CSV file", type=["csv"])

# --------------------
# Load Model and Tokenizer for Sentiment Classification
# --------------------
model_name = "taufiqdp/indonesian-sentiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

# --------------------
# Sentiment Classification Function
# --------------------
def classify_sentiment(tweet):
    if pd.isna(tweet):
        return None
    tweet = str(tweet)
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = torch.argmax(logits, dim=-1).item()
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[predicted_class]

# --------------------
# BERTopic Setup
# --------------------
def apply_bertopic(texts):
    # Initialize BERTopic and SentenceTransformer
    embedding_model = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')
    topic_model = BERTopic(language="indonesian", embedding_model=embedding_model)
    
    # Fit the model and transform the texts
    topics, probabilities = topic_model.fit_transform(texts)
    
    # Get Topic Information
    topic_info = topic_model.get_topic_info()
    return topic_model, topics, topic_info

# --------------------
# User Input for Sentiment Analysis (Persistent)
# --------------------
st.subheader("Enter a Tweet or Sentence for Sentiment Analysis")

# Input box for user sentence (persistent across reruns)
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input("Enter your tweet or sentence here:", value=st.session_state.user_input)

if user_input:
    # Save the user input to session state to persist it
    st.session_state.user_input = user_input

    # Classify the sentiment of the input sentence
    sentiment = classify_sentiment(user_input)
    
    # Display the result
    st.write(f"The sentiment of the input is: **{sentiment.capitalize()}**")

# --------------------
# Topic Modeling for New User Input (BERTopic)
# --------------------
st.subheader("Topic Modeling for Your Input (BERTopic)")

# Allow user to input multiple sentences
user_input_topic = st.text_area("Enter multiple sentences for Topic Modeling:")

if user_input_topic:
    # Apply BERTopic to the user's input
    st.subheader("Generated Topics")

    # Split input by new lines and apply BERTopic
    topic_model, topics, topic_info = apply_bertopic(user_input_topic.split('\n'))

    # Display Topics and the Top Words per Topic
    st.write(topic_info)

# --------------------
# Sentiment and Topic Analysis (For Uploaded File)
# --------------------
if uploaded_file:
    try:
        # Load Data
        data = pd.read_csv(uploaded_file)
        st.write("Data Loaded Successfully", data.head())

        # Check for required columns
        required_cols = ['Cleaned_Tweet_Content', 'updated_sentiment']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
            st.stop()

        # WordCloud for Tweets
        st.subheader("WordCloud of Tweets")
        all_text = " ".join(data['Cleaned_Tweet_Content'].dropna().astype(str))
        if all_text.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            st.image(wc.to_image())

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = data['updated_sentiment'].value_counts()
        st.write(sentiment_counts)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="viridis")
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Sentiment Classification using IndoBERT
        # if 'Cleaned_Tweet_Content' in data.columns:
        #     data['Sentiment_indobert'] = data['Cleaned_Tweet_Content'].apply(classify_sentiment)

        # Apply BERTopic
        if 'Cleaned_Tweet_Content' in data.columns:
            st.subheader("Topic Modeling (BERTopic)")

            # Apply BERTopic to the cleaned tweet content
            topic_model, topics, topic_info = apply_bertopic(data['Cleaned_Tweet_Content'].dropna().astype(str).tolist())

            filtered_topic_info = topic_info[(topic_info['Topic'] >= -1) & (topic_info['Topic'] <= 54)]

            # Show the filtered topic table
            st.write(filtered_topic_info)
            
            #st.write(topic_info)

            # Visualize Topic Distribution 
            st.subheader("Topic Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=filtered_topic_info,
                x="Topic",
                y="Count",
                palette="viridis",
                ax=ax
            )
            ax.set_title("Topic Distribution")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Count")
            plt.xticks(rotation=90)
            st.pyplot(fig)



    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty. Please check your file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
