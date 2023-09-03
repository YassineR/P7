#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
import json

from PIL import Image

st.set_page_config(
    page_title="Dashboard Credit Score Client",
    layout="wide" }
)

col11, col12 = st.columns([0.4,0.6])

with col11:
    image = Image.open('Logo.png')    
    st.image(image)

with col12:
    st.title('Prêt à dépenser')
    st.subheader("Scoring client")
    id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
    
