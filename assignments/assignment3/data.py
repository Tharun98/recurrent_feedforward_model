
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,optimizers
from keras.layers import Dense,Dropout,Activation,Flatten,LSTM
np.random.seed(7)


#---------------------------------------
def get_data(file_path):
    new_df=pd.read_csv(file_path)
    new_df=new_df.replace('null',np.nan)
    new_df=new_df.drop('Volume',axis=1)
    new_df=new_df.drop('Adj',axis=1)
    new_df=new_df.drop('Close.1',axis=1)
    new_df=new_df.dropna()
    new_df.to_csv('^mydata.csv')
    df=pd.read_csv('^mydata.csv')
    #print df
    # pattern
    '''
    previous days high,low
    5 days closing in order
    10,15,20,30 days moving avarage
    '''
    return_data=[]
    for i in range (len(df)-30):
        #print i
        present=df[i:i+31]
        tenth_sum=0
        fifteenth_sum=0
        twentieth_sum=0
        thirtieth_sum=0
        #print present
        high=present['High'][i+29]
        #print 'high',high
        low=present['Low'][i+29]
        close=present['Close'][25:i+31].real
        closing_array=present['Close']
        #if(i==0):
        #    print 'first next day closing price is ',nextday_close
        #print closing_array
        for k in range(30):
            ele=closing_array[i+k]
            thirtieth_sum=thirtieth_sum+ele
            if(k>=10):
                twentieth_sum=twentieth_sum+ele
            if(k>=15):
                fifteenth_sum=fifteenth_sum+ele
            if(k>=20):
                tenth_sum=tenth_sum+ele

        #if i==0:
        #    print [tenth_sum/10,fifteenth_sum/15,twentieth_sum/20,thirtieth_sum/30,nextday_close]
        array=[high,low]+[j for j in close]+[tenth_sum/10,fifteenth_sum/15,twentieth_sum/20,thirtieth_sum/30]
        #print array
        return_data.append(array)
    return return_data

def get_newdata(file_path):
    new_df=pd.read_csv(file_path)
    #print new_df
    new_df=new_df.replace('null',np.nan)
    new_df=new_df.drop('Volume',axis=1)
    new_df=new_df.drop('Adj',axis=1)
    new_df=new_df.drop('Close.1',axis=1)
    new_df=new_df.dropna()
    new_df.to_csv('^mynewdata.csv')
    df=pd.read_csv('^mynewdata.csv')
    #print df
    # pattern
    '''
    previous days high,low
    5 days closing in order
    10,15,20,30 days moving avarage
    '''
    return_data=[]
    for i in range (len(df)-30):
        #print i
        present=df[i:i+31]
        tenth_sum=0
        fifteenth_sum=0
        twentieth_sum=0
        thirtieth_sum=0
        #print present
        high=present['High'][i+29]
        #print 'high',high
        low=present['Low'][i+29]
        close=present['Close'][25:i+31].real
        closing_array=present['Close']
        #if(i==0):
        #    print 'first next day closing price is ',nextday_close
        #print closing_array
        for k in range(30):
            ele=closing_array[i+k]
            thirtieth_sum=thirtieth_sum+ele
            if(k>=10):
                twentieth_sum=twentieth_sum+ele
            if(k>=15):
                fifteenth_sum=fifteenth_sum+ele
            if(k>=20):
                tenth_sum=tenth_sum+ele

        #if i==0:
        #    print [tenth_sum/10,fifteenth_sum/15,twentieth_sum/20,thirtieth_sum/30,nextday_close]
        array=[high,low]+[j for j in close]+[tenth_sum/10,fifteenth_sum/15,twentieth_sum/20,thirtieth_sum/30]
        #print array
        return_data.append(array)
    return return_data

normalization_cofficients=[]                                            #for denormalization
def get_normalization_coefficents(df):
    for i in df.columns:
        m=max(df[i])
        n=min(df[i])
        normalization_cofficients.append([m,n])
    #return normalization_cofficients
def normalize(df):
    required_length=len(df.columns)
    count=0
    for i in df.columns:
        m=normalization_cofficients[count][0]
        n=normalization_cofficients[count][1]
        norm=0.15+(0.7*(df[i]-n)/(m-n))
        df[i]=norm
        count=count+1
    return df,required_length

#---------------------------------------------------------------
data=get_data('/home/dinesh/Desktop/stockmarket/assignments/assignment3/^BSESN.csv')
data_frame=pd.DataFrame(data,columns=['High','Low','Fifth','Fourhth','Third','Second','First','Nclose','Mtenth','Mfifteenth','Mtwentieth','Mthirtieth'])
#print data_frame
data_frame[data_frame.columns]=data_frame[data_frame.columns].applymap(np.int64)
data_frame.to_csv('^input.csv')
get_normalization_coefficents(data_frame)
norm_df,_=normalize(data_frame)
norm_df.to_csv('^norm.csv')
x=norm_df.drop('Nclose',axis=1)
input_data=x.as_matrix()
y=norm_df['Nclose']
output=[[i] for i in y.as_matrix()]
train_size=int(len(x))

#-------------------------------------------------------------------------
train_data=input_data[:train_size]
test_data=input_data[:]
train_output=output[:train_size]
test_output=output[:]

#------------------------------------------------------------
print 'before trainig normalization coefficients=',normalization_cofficients

model=Sequential()
model.add(Dense(14,input_dim=11,init='uniform',activation='tanh'))
#model.add(LSTM(11,input_dim=11,init='uniform',activation='sigmoid'))
model.add(Dense(29,activation='tanh'))
model.add(Dense(29,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
sgd = optimizers.SGD(lr=0.3, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['mean_squared_error'])
#model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(train_data,train_output,epochs=6,batch_size=1)
#-----------------------------------------------------------------
#testing


print 'testing'
scores=model.evaluate(test_data,test_output)
print 'scores',scores
#---------------------------------------------------------------------
#verifying

newdata=get_newdata('/home/dinesh/Desktop/stockmarket/assignments/assignment3/^BSESN_new.csv')
#print newdata
newdata_frame=pd.DataFrame(newdata,columns=['High','Low','Fifth','Fourhth','Third','Second','First','Nclose','Mtenth','Mfifteenth','Mtwentieth','Mthirtieth'])
newdata_frame[newdata_frame.columns]=newdata_frame[newdata_frame.columns].applymap(np.int64)
expclose=newdata_frame['Nclose']
newdata_frame.to_csv('^input_new.csv')
newnorm_df,required_length=normalize(newdata_frame)
newnorm_df.to_csv('^norm_new.csv')
#print 'required_length =',required_length
normalization_cofficients=normalization_cofficients[:required_length]
print 'before testing normalization coeffcients of next day close is',normalization_cofficients[7]

#print 'len of normalization cofficients = ',len(normalization_cofficients)
newx=newnorm_df.drop('Nclose',axis=1)
newinput_data=newx.as_matrix()
#newoutput=[[i] for i in newy.as_matrix()]

#print 'newinput_data',newinput_data[40]
#----------------------------------------------------------------------------
net_output=[]
for i in newinput_data:
    net_output.append(model.predict(np.array([[j for j in i]])))

net_output=np.array(net_output)
#net_output=model.predict(np.array([[i for i in newinput_data[40]]]))
#print 'net_output= ',net_output,'expeted normalized y= ',newy[40 ]

#-------------------------------------------------------------------
closing_coefficents=normalization_cofficients[7]
print 'coefficients used for normalization are ',closing_coefficents

m=closing_coefficents[0]
n=closing_coefficents[1]
denormalized_output=((net_output-0.15)/0.7)*(m-n)+n
denormalized_output=np.array([i[0][0] for i in denormalized_output]).astype(int)

#print 'predicted data=',denormalized_output
#print 'expected close =',expclose.as_matrix()
expclose_matrix=expclose.as_matrix()
sum_magnitudes=0
for i in range(len(denormalized_output)):
    sum_magnitudes=sum_magnitudes+abs(denormalized_output[i]-expclose_matrix[i])
avg_error=sum_magnitudes/len(denormalized_output)
#-----------------------------------------------------------------------------
print 'avg_error= ',avg_error
plt.plot(denormalized_output,'r')
plt.plot(expclose,'b')
plt.show()
