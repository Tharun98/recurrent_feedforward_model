import os
import pandas as pd

def get_data(file_path):
    df=pd.read_csv(file_path)
    # pattern
    '''
    previous days high,low
    5 days closing in order
    10,15,20,30 days moving avarage
    '''
    for i in range of (len(df)-30):
        present=df[i:i+30]
        thenth_sum=0
        fifteenth_sum=0
        twentieth_sum=0
        thirtieth_sum=0
        array=[]
        array.append(present['High'])
        
