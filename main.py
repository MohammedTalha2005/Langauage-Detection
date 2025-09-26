import streamlit as st
import pickle
from deep_translator import GoogleTranslator

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Language Detection & Translator", page_icon="🌍")

st.title("🌍 Language Detection & Translator App")
st.write("Enter text below, detect its language, and translate it into another language.")

# Input text
user_input = st.text_area("✍️ Enter text here", "")

# Language detection
if st.button("🔍 Detect Language"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        x = vectorizer.transform([user_input])
        prediction = model.predict(x)[0]
        st.success(f"✅ Detected Language: **{prediction}**")

# Translation Section
st.subheader("🌐 Translate Text")

# Allowed languages: Full name -> code
languages = {
    "English": "en",
    "Tamil": "ta",
    "Kannada": "kn",
    "Telugu": "te",
    "Malayalam": "ml",
    "Hindi": "hi"
}

# Dropdown with limited languages
target_lang_name = st.selectbox(
    "Select target language",
    list(languages.keys()),
    index=0
)

# Get language code from dictionary
target_lang_code = languages[target_lang_name]

if st.button("🌍 Translate"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to translate.")
    else:
        try:
            # Use your detector output as source language (better accuracy than auto)
            detected_lang = model.predict(vectorizer.transform([user_input]))[0]

            # Map detected language to translator codes
            lang_map = {
                "english": "en",
                "tamil": "ta",
                "kannada": "kn",
                "telugu": "te",
                "malayalam": "ml",
                "hindi": "hi"
            }

            source_code = lang_map.get(detected_lang.lower(), "auto")

            translated_text = GoogleTranslator(source=source_code, target=target_lang_code).translate(user_input)

            st.success(f"**Translated Text ({target_lang_name}):** {translated_text}")
        except Exception as e:
            st.error(f"❌ Translation failed: {e}")
