from SARIMAX_forecasting import SARIMAX
from anomalies_detect import anomaliesDetect
from plot_graph import plotAnomalies

date_time='DATE'
target='IPG2211A2N'

# train a forecasting model
forecasing = SARIMAX(r'Data/Electric_Production.csv')
df=forecasing.process_data(date_time,target)
train, train_log, test, test_log=forecasing.train_test_data(df, target)
predictions_df=forecasing.train_forecasting_model(df,train_log,test_log, date_time, len(test))

#detect anomalies
anomalies=anomaliesDetect()
df=anomalies.detect_classify_anomalies(predictions_df)

#plot anomalies
plot=plotAnomalies()
plot.plot_anomaly(df)
