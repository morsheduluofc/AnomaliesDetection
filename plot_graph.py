import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class plotAnomalies:
    def __init__(self):
          pass

    def plot_anomaly(self, df, title='Anomaly Detection'):

            bool_array = (abs(df['anomaly_points']) > 0)
            actuals = df['actuals'][-len(bool_array):]
            anomaly_points = bool_array * actuals
            anomaly_points[anomaly_points == 0] = np.nan
            
            anomalies = go.Scatter(name="Anomaly", x=df['date'],
                        xaxis='x1',
                        yaxis='y1',
                        y=df['anomaly_points'],
                        mode='markers',
                        marker = dict(color ='red',
                        size = 11,line = dict(
                                            color = "red",
                                            width = 2)))
            upper_bound = go.Scatter(hoverinfo="skip",
                            x=df['date'],
                            showlegend =False,
                            xaxis='x1',
                            yaxis='y1',
                            y=df['3std'],
                            marker=dict(color="#444"),
                            line=dict(
                                color=('rgb(23, 96, 167)'),
                                width=2,
                                dash='dash'),
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            fill='tonexty')
            lower_bound = go.Scatter(name='Confidence Interval',
                            x=df['date'],
                            xaxis='x1',
                            yaxis='y1',
                            y=df['-3std'],
                            marker=dict(color="#444"),
                            line=dict(
                                color=('rgb(23, 96, 167)'),
                                width=2,
                                dash='dash'),
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            fill='tonexty')
            
            Error = go.Scatter(name='Error',
                    x=df['date'], y=df['error'],
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(size=12,
                                line=dict(width=1),
                                color="red"),
                    text="Error")

            Mvingavrg = go.Scatter(name='Moving Average',
                            x=df['date'],
                            y=df['mean_val'],
                            xaxis='x1',
                            yaxis='y1',
                            marker=dict(size=12,
                                        line=dict(width=1),
                                        color="green"),
                            text="Moving average")

            Actuals = go.Scatter(name= 'Actuals',
                        x= df['date'],
                        y= df['actuals'],
                        xaxis='x2', yaxis='y2',
                        marker=dict(size=12,
                                    line=dict(width=1),
                                    color="blue"))
            
            Predicted = go.Scatter(name= 'Predicted',
                        x= df['date'],
                        y= df['predicted'],
                        xaxis='x2', yaxis='y2',
                        marker=dict(size=12,
                                    line=dict(width=1),
                                    color="orange"))
            
            anomalies_map = go.Scatter(name = "anomaly actual",
                                    showlegend=False,
                                    x=df['date'],
                                    y=anomaly_points,
                                    mode='markers',
                                    xaxis='x2',
                                    yaxis='y2',
                                    marker = dict(color ="red",
                                    size = 11,
                                    line = dict(color = "red", width = 2)))

            color_map= {0: "lightgray", 1: "yellow", 2: "orange", 3: "red"}
            df.sort_values(by='date', ascending = True, inplace=True)
            table = go.Table(
                domain=dict(x=[0, 1], y=[0, 0.3]),
                columnwidth=[1, 2 ],
                #columnorder=[0, 1, 2,],
                header = dict(height = 20,
                            values = [['<b>Date</b>'],['<b>Actual Values </b>'],
                                        ['<b>Predicted</b>'], ['<b>% Difference</b>'],['<b>Severity (0-3)</b>']],
                            font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                            fill = dict(color='#d562be')),
                cells = dict(values = [df.round(3)[k].tolist() for k in ['date', 'actuals', 'predicted','error_percentage','color']],
                            line = dict(color='#506784'),
                            align = ['center'] * 5,
                            font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                            #format = [None] + [",.4f"] + [',.4f'],#suffix=[None] * 4,
                            suffix=[None] + [''] + [''] + ['%'] + [''],
                            height = 27,
                            #fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']))
                            fill=dict(color=  # ['rgb(245,245,245)',#unique color for the first column
                                [df['color'].map(color_map)],
                                )
                ))
                
            axis=dict(
                showline=True,
                zeroline=False,
                showgrid=True,
                mirror=True,
                ticklen=4,
                gridcolor='#ffffff',
                tickfont=dict(size=10))
                
            layout = dict(
                width=1000,
                height=865,
                autosize=False,
                title= title,
                margin = dict(t=75),
                showlegend=True,
                xaxis1=dict(axis, **dict(domain=[0, 1], anchor='y1', showticklabels=True)),
                xaxis2=dict(axis, **dict(domain=[0, 1], anchor='y2', showticklabels=True)),
                yaxis1=dict(axis, **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor='x1', hoverformat='.2f')),
                yaxis2=dict(axis, **dict(domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor='x2', hoverformat='.2f')))

            fig = go.Figure(data = [anomalies,anomalies_map,upper_bound,lower_bound,Actuals,Predicted, Mvingavrg,Error, table],layout=layout)
            #iplot(fig)
            #pyplot.show()
            fig.show()
