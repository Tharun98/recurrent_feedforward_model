import os
import pandas as pd

def get_data(file_path):
    df=pd.read_csv(file_path)
    df=df[0:31]
    # pattern
    '''
    previous days high,low
    5 days closing in order
    10,15,20,30 days moving avarage
    '''
    for i in range (len(df)-30):
        present=df[i:i+30]
        tenth_sum=0
        fifteenth_sum=0
        twentieth_sum=0
        thirtieth_sum=0
        array=[]
        high=present['High'][29:30]
        print high
        low=present['Low'][29:30]
        close=present['Close'][25:30]
        closing_array=present['Close']
        for i in range(30):
            ele=eval(closing_array[i])
            thirtieth_sum=thirtieth_sum+ele
            if(i>=20):
                twentieth_sum=twentieth_sum+ele
            if(i>=15):
                fifteenth_sum=fifteenth_sum+ele
            if(i>=10):
                tenth_sum=tenth_sum+ele
        array=array+[high,low,close,tenth_sum/10,fifteenth_sum/15,twentieth_sum/20,thirtieth_sum/30]
        print array

get_data('/home/dinesh/Desktop/projects/recurrent_feedforward_model/assignments/assignment3/^BSESN.csv'
)
