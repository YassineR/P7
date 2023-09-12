import shap 
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.subplots as sp
import plotly.graph_objs as go

px.defaults.color_continuous_scale = px.colors.sequential.Darkmint

def get_plot_global_feature_importance(model, columns ):
    importance_df = pd.DataFrame(dict(
    group = columns,
    value = model.feature_importances_))

    importance_df.sort_values('value',ascending=False, inplace = True)

    importance_df = importance_df[0:20]

    importance_df.sort_values('value', inplace = True)
    
    fig = px.bar(importance_df, y = 'group', x = 'value',
                 color = 'value')
    return fig

def get_plot_local_feature_importance(model, client_row):
    
    local_importance = model.predict(client_row, pred_contrib=True)[0][0:-1]

    # Sample data
    importance_df = pd.DataFrame(dict(
        group = client_row.columns,
        value = local_importance))

    importance_df.sort_values('value',ascending=False, inplace = True)

    importance_df = importance_df[0:20]

    importance_df.sort_values('value', inplace = True)

    fig = px.bar(importance_df, y = 'group', x = 'value',
                 color = 'value')
    #fig. update_xaxes( autorange="reversed")
    fig. update_yaxes( side = 'right')
    fig.update_coloraxes(colorbar_x=-0.2)
    
    return fig

def barplot_client_majority(df, client_data):    
        
    if(len(client_data) <= 0):
        return
    
    # Calculate the features of interest for the selected client
    client_amt_credit = client_data['AMT_CREDIT'].values[0]
    client_amt_annuity = client_data['AMT_ANNUITY'].values[0]
    client_amt_goods_price = client_data['AMT_GOODS_PRICE'].values[0]

    # Calculate the mean values for the majority age group
    majority_amt_credit = np.round(df['AMT_CREDIT'].mean(), decimals=2) 
    majority_amt_annuity = np.round(df['AMT_ANNUITY'].mean(), decimals=2) 
    majority_amt_goods_price = np.round(df['AMT_GOODS_PRICE'].mean(), decimals=2) 

    # Create subplots for each feature comparison
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=(
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"))


    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], y=[client_amt_credit, majority_amt_credit],
                  name='AMT_CREDIT', text=[client_amt_credit, majority_amt_credit]), row=1, col=1)

    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], 
                         y=[client_amt_annuity, majority_amt_annuity], name='AMT_ANNUITY', 
                         text=[client_amt_annuity, majority_amt_annuity]), row=1, col=2)

    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], 
                         y=[client_amt_goods_price, majority_amt_goods_price], name='AMT_GOODS_PRICE', 
                         text=[client_amt_goods_price, majority_amt_goods_price]), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'Comparaison du client{client_id_to_compare} à l\'ensembe des clients')

    return fig

def barplot_same_clients(df, client_data):    
    
    if(len(client_data) <= 0):
        return
    
    # Calculate the features of interest for the selected client
    client_amt_credit = client_data['AMT_CREDIT'].values[0]
    client_amt_annuity = client_data['AMT_ANNUITY'].values[0]
    client_amt_goods_price = client_data['AMT_GOODS_PRICE'].values[0]
    
    
    
    client_EXT_SOURCE_3 = client_data['EXT_SOURCE_3'].values[0]
    client_EXT_SOURCE_2 = client_data['EXT_SOURCE_2'].values[0]
    client_PAYMENT_RATE = client_data['PAYMENT_RATE'].values[0]

    # Calculate the mean values for the majority age group
    
    same_clients = df[
        (df['AMT_CREDIT'] <= client_amt_credit*1.10)  & (df['AMT_CREDIT'] >= client_amt_credit*0.90)
        & (df['AMT_ANNUITY'] <= client_amt_annuity*1.10) & (df['AMT_ANNUITY'] >= client_amt_annuity*0.90)
        & (df['AMT_GOODS_PRICE'] <= client_amt_goods_price*1.10)  & (df['AMT_GOODS_PRICE'] >= client_amt_goods_price*0.90)        
        
    ]#[['AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']]
    print(same_clients.shape)
    
    mean_same_client_amt_credit = np.round(same_clients['EXT_SOURCE_3'].mean(), decimals=2) 
    mean_same_client_amt_annuity = np.round(same_clients['EXT_SOURCE_2'].mean(), decimals=2) 
    mean_same_client_goods_price = np.round(same_clients['PAYMENT_RATE'].mean(), decimals=2) 

    # Create subplots for each feature comparison
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=(
        "EXT_SOURCE_3", "EXT_SOURCE_2", "PAYMENT_RATE"))


    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], y=[client_EXT_SOURCE_3, mean_same_client_amt_credit],
                  name='EXT_SOURCE_3', text=[client_EXT_SOURCE_3, mean_same_client_amt_credit]), row=1, col=1)

    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], 
                         y=[client_EXT_SOURCE_2, mean_same_client_amt_annuity], name='EXT_SOURCE_2', 
                         text=[client_EXT_SOURCE_2, mean_same_client_amt_annuity]), row=1, col=2)

    fig.add_trace(go.Bar(x=['Client', 'Moyenne'], 
                         y=[client_PAYMENT_RATE, mean_same_client_goods_price], name='PAYMENT_RATE', 
                         text=[client_PAYMENT_RATE, mean_same_client_goods_price]), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'Comparaison du client{client_id_to_compare} à l\'ensembe des clients')
    
    return fig

