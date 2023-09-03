#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
import json

from PIL import Image

st.set_page_config(
    page_title="Dashboard Credit Score Client",
    layout="wide" 
)

def update_ext_source_3(*args):
    st.session_state.test = st.session_state.ext_source_3

st.session_state.ext_source_3 = 0.57 if 'ext_source_3' not in st.session_state else st.session_state.ext_source_3
col11, col12 = st.columns([0.4,0.6])

with col11:
    image = Image.open('Logo.png')    
    st.image(image)

with col12:
    st.title('Prêt à dépenser')
    st.subheader("Scoring client")
    id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
    
    st.number_input('Minutes', key='test', min_value=5, max_value=25, step=1, on_change=update_base)
    varTest = 0.56
    varTest = st.number_input('EXT_SOURCE_3',
                                 min_value=0., value=varTest, step=1., max_value = 1, on_change=update_ext_source_3)
    
