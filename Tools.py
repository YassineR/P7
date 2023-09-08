import shap 
import pandas as pd
import plotly.express as px

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
    fig.update_coloraxes(colorbar_orientation='h')
    
    return fig
