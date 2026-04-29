import streamlit as st
import requests

st.set_page_config(page_title="Mamba Sentiment Analyzer", page_icon="🐍")
st.title("🐍 Mamba Sentiment Analysis")
st.write("Cette interface interroge ton API FastAPI tournant dans Docker.")

text_input = st.text_area("Entrez une critique de film :", "I loved this movie, the acting was great!")

if st.button("Analyser le sentiment"):
    payload = {"text": text_input}
    try:
        response = requests.post("http://api:8000/predict", json=payload)
        data = response.json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", data["sentiment"])
        with col2:
            st.metric("Confiance", data["confidence"])
            
        if data["sentiment"] == "Positif":
            st.success("L'IA pense que c'est un avis positif !")
        else:
            st.error("L'IA pense que c'est un avis négatif !")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
