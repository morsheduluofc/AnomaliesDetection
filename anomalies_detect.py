import numpy as np

class anomaliesDetect:
    def __init__(self):
        pass
    
    #detect and classify the anomalies
    def detect_classify_anomalies(self,df, window=5):
            df.replace([np.inf,-np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            df['error']=df['actuals']-df['predicted']
            df['error_percentage']=((df['actuals']-df['predicted'])/df['actuals'])*100

            df['mean_val']=df['error'].rolling(window=window).mean()
            df['std_val']=df['error'].rolling(window=window).std()
            df['-3std']=df['mean_val']-(2*df['std_val'])
            df['3std']=df['mean_val']+(2*df['std_val'])
            df['-2std']=df['mean_val']-(1.75*df['std_val'])
            df['2std']=df['mean_val']+(1.75*df['std_val'])
            df['-1std']=df['mean_val']-(1.5*df['std_val'])
            df['1std']=df['mean_val']-(1.5*df['std_val'])
            df.reset_index(drop = True,inplace=True)
            cut_list=df[['error', '-3std', '-2std', '-1std', 'mean_val', '1std', '2std', '3std']]
            cut_values=cut_list.values
            cut_sort=np.sort(cut_values)
            #print([x for x in range(len(df['error']))])

            df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in range(len(df['error']))]
            severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
            region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE", 7: "POSITIVE"}
            df['color'] =  df['impact'].map(severity)
            df['region'] = df['impact'].map(region)
            df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)
            df = df.sort_values(by='date', ascending=False)
            #df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y-%m-%d")
            return df