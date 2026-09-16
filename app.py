import streamlit as st
import joblib

# Load the saved model (a Pipeline with TF-IDF + classifier already inside it)
with open("best_sentiment_model.joblib", "rb") as f:
    model = joblib.load(f)

st.title("Review Sentiment Classifier")
st.write("Enter a review summary to predict its sentiment.")

text = st.text_area("Review Summary", placeholder="e.g. Absolutely loved this product!")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([text])[0]
        st.subheader(f"Predicted Sentiment: {prediction}")
