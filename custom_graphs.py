import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)


def model_performance(model, y_test, y_preds, y_score):
    
    #Conf matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_preds)
    trace1 = go.Heatmap(z=conf_matrix, x=["0.0 (pred)", "1.0 (pred)"],
                        y=["0.0 (true)", "1.0 (true)"], xgap=4, ygap=4,
                        colorscale='GnBu', showscale=False)
    
    #Show metrics
    show_metrics = pd.DataFrame(data=[[metrics.accuracy_score(y_test, y_preds),
                                       metrics.balanced_accuracy_score(y_test, y_preds),
                                       metrics.precision_score(y_test, y_preds), 
                                       metrics.recall_score(y_test, y_preds), 
                                       metrics.f1_score(y_test, y_preds)]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'white']
    trace2 = go.Bar(x=(show_metrics[0].values), 
                    y=['Accuracy', 'Balanced<br>accuracy', 'Precision', 'Recall', 'F1_score'], 
                    text=np.round_(show_metrics[0].values, 4),
                    textposition='auto',
                    orientation='h', opacity=0.8, marker=dict(color=colors,
                                                              line=dict(color='#000000', width=1.5)))
    
    #Roc curve
    model_roc_auc = round(metrics.roc_auc_score(y_test, y_score), 3)
    fpr, tpr, t = metrics.roc_curve(y_test, y_score)
    trace3 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color=('rgb(22, 96, 167)'), width=2), fill='tozeroy')
    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('black'), width=1.5,
                        dash='dot'))
    
    # Precision-recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
    trace5 = go.Scatter(x=recall, y=precision,
                        name = "Precision" + str(precision),
                        line = dict(color=('lightcoral'), width=2), fill='tozeroy')
    
    
    #Subplots
    fig = make_subplots(rows=2, cols=2, print_grid=False, 
                        specs=[[{}, {}], 
                               [{}, {}]],
                        subplot_titles=('Confusion Matrix',
                                        'Metrics',
                                        'ROC curve' + " " + '(' + str(model_roc_auc) + ')',
                                        'Precision - Recall curve')
                       )
    
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)
    
    fig["layout"].update(showlegend=False, title={'text': '<b>Model performance report</b><br>' + str(model) \
                                                  + '<br>' * 27 \
                                                  + str(metrics.classification_report(y_test, y_preds)).replace('\n', '<br>'),
                                                  'x': 0.5,
                                                  'y': 0.95,
                                                  'xanchor': 'center',
                                                  'yanchor': 'top'},
                         autosize=False, height=800, width=900,
                         plot_bgcolor='rgba(240, 240, 240, 0.95)',
                         paper_bgcolor='rgba(240, 240, 240, 0.95)',
                         margin=dict(b=270))
    fig["layout"]["xaxis1"].update(dict(title='<i>True label</i>'))
    fig["layout"]["yaxis1"].update(dict(title='<i>Predicted label</i>'))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title='<i>false positive rate</i>'))
    fig["layout"]["yaxis3"].update(dict(title='<i>true positive rate</i>'))
    fig["layout"]["xaxis4"].update(dict(title='<i>recall</i>'), range=[0, 1.05])
    fig["layout"]["yaxis4"].update(dict(title='<i>precision</i>'), range=[0, 1.05])
    
    fig.layout.titlefont.size = 16
        
    py.iplot(fig)


def conversion_log_comparison(df1, column1, df2, column2):
    
    fig = make_subplots(rows=3, cols=2, row_heights=[0.2, 0.35, 0.45],
                        subplot_titles=('<b>' + column1 + ' variable distribution</b>',
                                        '<b>nLog ' + column1 + ' variable distribution</b>'),
                        vertical_spacing=0.02,
                        horizontal_spacing=0.035)

    fig.add_trace(px.box(df1, 
                         x=column1, 
                         orientation='h',
                         color_discrete_sequence=['#393E46'])
                  .data[0], 
                  row=1, col=1)

    fig.add_trace(go.Histogram(x=df1[column1], 
                               opacity=1,
                               nbinsx=55,
                               marker_color='#393E46',
                               hovertemplate='<extra>' + column1 + ': %{x}<br>count: %{y}</extra>',
                               showlegend=False),
                  row=2, col=1)

    fig.add_trace(px.ecdf(df1, 
                          x=column1, 
                          markers=True, 
                          lines=False,
                          color_discrete_sequence=['#393E46'])
                  .data[0], 
                  row=3, col=1)

    fig.add_trace(px.box(df2, 
                         x=column2, 
                         orientation='h', 
                         color_discrete_sequence=['#F66095'])
                  .data[0], 
                  row=1, col=2)

    fig.add_trace(go.Histogram(x=df2[column2], 
                               opacity=1, 
                               nbinsx=50,
                               marker_color='#F66095',
                               hovertemplate='<extra>' + column2 + ': %{x}<br>count: %{y}</extra>',
                               showlegend=False), 
                  row=2, col=2)

    fig.add_trace(px.ecdf(df2, 
                          x=column2, 
                          markers=True, 
                          lines=False, 
                          color_discrete_sequence=['#F66095'])
                  .data[0], 
                  row=3, col=2)

    fig['layout']['yaxis3']['title'] = '<i>count</i>'
    fig['layout']['yaxis5']['title'] = '<i>probability</i>'
    fig['layout']['xaxis1']['showticklabels'] = False
    fig['layout']['xaxis2']['showticklabels'] = False
    fig['layout']['xaxis3']['showticklabels'] = False
    fig['layout']['xaxis4']['showticklabels'] = False

    fig['layout']['xaxis5']['title'] = '<i>' + column1.lower() + '</i>'
    fig['layout']['xaxis6']['title'] = '<i>' + column2.lower() + '</i>'

    fig.update_layout(
        width=980, 
        height=700,
        margin=dict(l=0, r=0, t=40, b=40)
    )

    py.iplot(fig)