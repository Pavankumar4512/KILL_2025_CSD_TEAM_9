import streamlit as st
import wikipedia
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load FAQ and Small Talk Data
def load_json(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

small_talk = load_json("small_talk.json")
faqs = load_json("faqs.json")

# Extract FAQ questions and answers
questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# TF-IDF Vectorizer for FAQs
vectorizer = TfidfVectorizer()
if questions:
    question_vectors = vectorizer.fit_transform(questions)
else:
    question_vectors = None

# Wikipedia Search Function
def get_wikipedia_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:3])}. Please be more specific."
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find any information."
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Main Chatbot Response Function
def get_best_response(user_input):
    user_input_lower = user_input.lower().strip()

    # Small Talk Handling
    for qa in small_talk:
        if user_input_lower == qa["question"].lower():
            return qa["answer"]

    # FAQ Matching
    if question_vectors is not None:
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, question_vectors).flatten()
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] > 0.3:
            return answers[best_match_idx]

    # Wikipedia Search
    wiki_response = get_wikipedia_summary(user_input)
    if "error" not in wiki_response.lower():
        return wiki_response

    return "I'm sorry, I couldn't find an answer."

# Streamlit UI Styling
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextInput, .stTextArea {
            border-radius: 10px;
            background-color: #333;
            color: white;
        }
        .branding {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 16px;
            color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Branding
st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Ask me anything and I'll try my best to help!</p>", unsafe_allow_html=True)

# Chat UI
def chat_ui():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.text_area("You:", "", key="user_input")

    if st.button("Send", use_container_width=True):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = get_best_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

chat_ui()

# Branding at the bottom right corner
st.markdown("""
    <div class='branding'>
        <strong>Establisher:</strong> Pavan kumar Bantu<br>
        <strong>Contact:</strong> pavanbantu2005@gmail.com
    </div>
""", unsafe_allow_html=True)
