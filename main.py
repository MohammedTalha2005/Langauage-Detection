import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Language Detection App", page_icon="🌍")

st.title("🌍 Language Detection App")
st.write("Enter text below and the model will detect the language.")

# Input text
user_input = st.text_area("Enter text here", "")

if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Transform input
        x = vectorizer.transform([user_input])
        prediction = model.predict(x)[0]
        
        st.success(f"✅ Detected Language: **{prediction}**")
