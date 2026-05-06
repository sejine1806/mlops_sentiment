import streamlit as st
import requests

st.set_page_config(page_title="Mamba Sentiment Analyzer", page_icon="🐍")
st.title("🐍 Mamba Sentiment Analysis")
st.write("This interface queries your FastAPI running in Docker.")

text_input = st.text_area("Submit a movie review :", "I loved this movie, the acting was great!")

if st.button("Analyze sentiment"):
    payload = {"text": text_input}
    try:
        response = requests.post("http://api:8000/predict", json=payload)
        data = response.json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", data["sentiment"])
        with col2:
            st.metric("Confiance", data["confidence"])
            
        if data["sentiment"] == "Positiv":
            st.success("Positiv review !")
        else:
            st.error("Negativ review !")
    except Exception as e:
        st.error(f"Connection error to the API : {e}")
