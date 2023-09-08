import shap 

def get_plot_global_feature_importance(model ):
    importance_df = pd.DataFrame(dict(
    group = valid_x.columns,
    value = model.feature_importances_))

    importance_df.sort_values('value',ascending=False, inplace = True)

    importance_df = importance_df[0:20]

    importance_df.sort_values('value', inplace = True)
    
    fig = px.bar(importance_df, y = 'group', x = 'value',
                 color = 'value')
    return fig

def get_plot_local_feature_importance(client_row):
    
    local_importance = lgbm_clf.predict(client_row, pred_contrib=True)[0][0:-1]

    # Sample data
    importance_df = pd.DataFrame(dict(
        group = valid_x.columns,
        value = local_importance))

    importance_df.sort_values('value',ascending=False, inplace = True)

    importance_df = df3[0:20]

    importance_df.sort_values('value', inplace = True)

    fig = px.bar(importance_df, y = 'group', x = 'value',
                 color = 'value')
    #fig. update_xaxes( autorange="reversed")
    fig. update_yaxes( side = 'right')
    fig.update_coloraxes(colorbar_orientation='h')
    
    
    fig.show()