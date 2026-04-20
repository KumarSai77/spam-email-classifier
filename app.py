import streamlit as st
import pickle

# Load model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Classifier")

input_text = st.text_area("Enter your message")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter a message")
    else:
        vec = vectorizer.transform([input_text])
        result = model.predict(vec)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")