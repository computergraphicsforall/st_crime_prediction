import plotly.graph_objects as go


def forecast_graphs(dt, de, pt, pe, dc, ffe, fip, title, c):
    """
    Function to plot the prediction results

    :param dt: dataframe with the full data filterer between start and end date
    :param de: training dataframe filtered between start and end date
    :param pt: trend forecast results
    :param pe: seasonal forecast results
    :param dc: decomposition of training time series
    :param ffe: training end date
    :param fip: prediction start date
    :param title: chart title
    :param c: data composition: True for when there is composition. Otherwise False
    :return: a graph with the prediction, original data, training and decomposition of the time series
    """
    fig = go.Figure()
    fig.layout.template = 'plotly_dark'
    fig.layout.yaxis.fixedrange = False
    fig.layout.xaxis.fixedrange = True

    fig.add_trace(go.Scatter(x=dt.index, y=dt.EVENTS, mode='lines', name='Origin', line=dict(color='#8aff8c')))

    if c:
        # decomposition time series
        fig.add_trace(go.Scatter(x=de.index, y=dc.seasonal, mode='lines', name='Season'))
        fig.add_trace(go.Scatter(x=de.index, y=dc.resid, mode='lines', name='Resid'))
        fig.add_trace(
            go.Scatter(x=[ffe, fip], y=[dc.trend[len(dc.trend) - 1], pt.drift[0]], mode='lines', line=dict(color='red'),
                       showlegend=False))
        fig.add_trace(go.Scatter(x=de.index, y=dc.trend, mode='lines', name='Trend'))

    else:
        # decomposition time series
        fig.add_trace(go.Scatter(x=de.index, y=dc.seasonal.EVENTS, mode='lines', name='Season'))
        fig.add_trace(go.Scatter(x=de.index, y=dc.resid.EVENTS, mode='lines', name='Resid'))
        fig.add_trace(go.Scatter(x=[ffe, fip], y=[dc.trend.EVENTS[len(dc.trend.EVENTS) - 1], pt.drift[0]], mode='lines',
                                 line=dict(color='red'), showlegend=False))
        fig.add_trace(go.Scatter(x=de.index, y=dc.trend.EVENTS, mode='lines', name='Trend'))

    # trend
    fig.add_trace(go.Scatter(x=pt.index, y=pt.drift, mode='lines+markers', name='Trend-fcast', line=dict(color='red')))
    # forecast
    fig.add_trace(go.Scatter(x=pe.index, y=pe['drift+seasonal'], mode='lines+markers', name='Predict',
                             line=dict(color='#8accff', width=1, dash='dot')))

    fig.update_layout(title='%s' % ("<b>" + title + "</b>"), dragmode='pan',
                      margin={'l': 50, 'r': 100, 'b': 50, 't': 100}, width=900, height=500,
                      xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_xaxes(range=[de.index[0], pt.index[len(pt) - 1]])
    # fig.write_image("images/"+title+".svg")
    fig.show(config=dict({'scrollZoom': True}))
