#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
import json

import pickle
from PIL import Image
from lightgbm import LGBMClassifier
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import rgb2hex
import plotly.graph_objs as go

st.set_page_config(
    page_title="Dashboard Credit Score Client",
    layout="wide" 
)

def set_color_range(probability):
    cmapR = cm.get_cmap('RdYlGn')
    norm = Normalize(0, 1)
    return rgb2hex(cmapR(norm(probability)))
    

@st.cache_resource 
def load_model():
    lgbm = pickle.load(open('model.pkl', 'rb'))
    return lgbm

@st.cache_data #mise en cache de la fonction pour exécution unique
def load_dataframe():
    data = pd.read_csv('data.csv')
    return data


data = load_dataframe()
lgbm = load_model()

def lgbm_prediction(_data, _id_client, _model):
    feats = [f for f in _data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    _data = _data[data["SK_ID_CURR"] == st.session_state.index]
    if(_data.shape[0]==0):
        return -1
    else:
        return  _model.predict_proba(_data[feats])[0][0]
    



# def update_ext_source_3(*args):
#     st.session_state.test = st.session_state.ext_source_3

def update_index(*args):
    pred = lgbm_prediction(data, st.session_state.index, lgbm)
    st.session_state.prediction = pred
    


st.session_state.ext_source_3 = 0.57 if 'ext_source_3' not in st.session_state else st.session_state.ext_source_3
st.session_state.index = 0 if 'index' not in st.session_state else st.session_state.index
st.session_state.prediction = 0. if 'prediction' not in st.session_state else st.session_state.prediction

col11, col12 = st.columns([0.4,0.6])

with col11:
    image = Image.open('Logo.png')    
    st.image(image)

with col12:
    st.title('Prêt à dépenser')
    st.subheader("Scoring client")
    id_input = st.number_input('Veuillez saisir l\'identifiant d\'un client:',key = 'index',min_value = 0, on_change=update_index )

    if(st.session_state.prediction == -1):
        st.session_state.text =  "Client introuvable "+ str(type(st.session_state.index)) + "  --  " + str(st.session_state.index)
        st.text(body = st.session_state.text  )

    else:        
        color = set_color_range(st.session_state.prediction)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            number={'suffix': "%"},
            value= st.session_state.prediction*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Client sûr"},
            gauge={'axis': {'range': [None, 100]},
                   'bar' : {'color': color}
                  },
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    
    st.number_input('EXT_SOURCE_3', key = 'ext_source_3',
                                 min_value=0., step=0.1, max_value = 1.)
    
    st.number_input('Prediction', key='prediction')

    
    st.dataframe(data.head())
    
