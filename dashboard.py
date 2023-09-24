#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
from requests.exceptions import ConnectionError
import Tools

import pickle
from PIL import Image
from lightgbm import LGBMClassifier
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import rgb2hex
import plotly.graph_objs as go
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

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
        
    try:
        response = requests.get('http://yrp7.azurewebsites.net/get-model')
    except ConnectionError as e:    # This is the correct syntax
        response = "No response"

    if response !="No response":
        if response.status_code == 200:
            # Deserialize the received model
            lgbm = pickle.loads(response.content)
            st.session_state.api_status = 'OK'
            
    else:
        print("Failed to retrieve the model from the API")
        lgbm = pickle.load(open('model.pkl', 'rb'))
        st.session_state.api_status = 'NOK'
        
    return lgbm

@st.cache_data #mise en cache de la fonction pour exécution unique
def load_dataframe():
    data = pd.read_csv('home_credit_data_sample.csv')
    return data

@st.cache_data #mise en cache de la fonction pour exécution unique
def set_feats(_data):    
    feats = [f for f in _data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    return feats
    


data = load_dataframe()
lgbm = load_model()
feats = set_feats(data)

def lgbm_prediction(_data, _id_client, _model):
    #feats = [f for f in _data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    _data = _data[data["SK_ID_CURR"] == st.session_state.index]
    st.session_state.row = _data 
    if(_data.shape[0]==0):
        return -1
    else:
        return  _model.predict_proba(_data[feats])[0][0]
    

def update_index(*args):
    pred = lgbm_prediction(data, st.session_state.index, lgbm)
    st.session_state.prediction = pred
    


st.session_state.api_status = 'OK' if 'api_status' not in st.session_state else st.session_state.api_status
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

    
    if(st.session_state.api_status == 'NOK'):  
        st.warning('Model successfully loaded locally, but the API is currently unavailable.', icon="⚠️")
        
    if(st.session_state.prediction == -1):        
        st.session_state.text =  "Client introuvable "
        st.error(st.session_state.text )
        
        # st.text(body = st.session_state.text  )

    elif(st.session_state.prediction > 0 ): 

        col121, col122 = st.columns([0.4,0.6])

        with col121:
            st.text('\n\n\n')
            st.subheader("Informations clients")
            
            st.text('')
            st.text('')

            
            gender = 'Homme' if st.session_state.row['CODE_GENDER'].values[0] == 0 else 'Femme'  
            st.text(body = 'Genre : ' + gender)
            
            st.text(body = 'Montant du produit : ' + str(st.session_state.row['AMT_GOODS_PRICE'].values[0] )+ ' $')
            st.text(body ='Montant du crédit : ' + str(st.session_state.row['AMT_CREDIT'].values[0] ) + ' $')
            st.text(body = 'Montant de l\'annuité : ' + str(st.session_state.row['AMT_ANNUITY'].values[0] )+ ' $')
            
            age = int(np.floor((-1*st.session_state.row['DAYS_BIRTH'].values[0]) / 365))
            st.text(body = 'Age : '+ str(age) + ' ans' )

            indication = 'Accepté' if st.session_state.prediction > 0.5 else 'Refusé'         
            st.text(body = 'Indication  : '+ indication )

        
            
        with col122:
        
            color = set_color_range(st.session_state.prediction)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                number={'suffix': "%"},
                value= st.session_state.prediction*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Client sûr"},
                gauge={'axis': {'range': [None, 100]},
                       'bar' : {'color': color}
                      },
            ))
        
            st.plotly_chart(gauge_fig, use_container_width=True)
if(st.session_state.prediction > 0 ):        
    
    tab1, tab2, tab3 = st.tabs(["Importance des features", "Ensemble des clients", "Clients similaires"])

    global_FI_fig = Tools.get_plot_global_feature_importance(lgbm,st.session_state.row[feats].columns)
    local_FI_fig = Tools.get_plot_local_feature_importance(lgbm, st.session_state.row[feats])
    barplot_client_majority = Tools.barplot_client_majority(data, st.session_state.row)
    barplot_same_clients = Tools.barplot_same_clients(data, st.session_state.row)
    
    with tab1:
        col21, col22 = st.columns(2)
    
        with col21:
            st.subheader("        Local feature importance")
            st.text(body = "        Informations impactant le plus le client" )
            st.plotly_chart(local_FI_fig, use_container_width=True)
            
        with col22:        
            st.subheader("        Global feature importance")
            st.text(body = "        Impact moyen sur les clients " )
            st.plotly_chart(global_FI_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Information du client vs Moyenne ")
        st.plotly_chart(barplot_client_majority, use_container_width=True)
    
    with tab3:
        st.subheader("Information du client vs Clients similaires ")
        st.text(body = "Les clients similaires rassemblent les clients ayant le mêmes crédit, annuité et valeur du produit (+/- 10%)" )
        st.plotly_chart(barplot_same_clients, use_container_width=True)       
        
