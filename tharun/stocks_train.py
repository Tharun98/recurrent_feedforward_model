import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from keras.models import Sequential,optimizers
from keras.layers import Dense,Dropout,Activation,Flatten,LSTM
t=pd.read_csv('~/^BSESN.csv',nrows = 5000)
#q=pd.read_csv('~/SENSEX.csv',nrows = 5000)
x = t.Close[0:len(t)]
y = t.High[0:len(t)]
z = t.Low[0:len(t)]
index, value = max(enumerate(x), key=operator.itemgetter(1))
index1, value1 = max(enumerate(y), key=operator.itemgetter(1))
index2, value2 = max(enumerate(z), key=operator.itemgetter(1))
count = 0
'''print "index is %d,value is %f" %(index,value)
print "index1 is %d,value1 is %f" %(index1,value1)
print "index2 is %d,value2 is %f" %(index2,value2)'''
x = x/x[index]         # final array of closing values
y = y/y[index1]        # final array of high values
z = z/z[index2]        #final values of low values
print y[30]
output_array = x[31:4904]
for i in range(30,4903) :              #taking input data from row 30 till 4903
 count = count +1
 input_array = []    
 High1 = y[i]   
 Low1 = z[i]
 j = i - 4 ;
 Close1 = x[j]
 Close2 = x[j+1]
 Close3 = x[j+2]
 Close4 = x[j+3]
 Close5 = x[j+4]
 f = 0
 average=0
 a1=0
 average1=0
 b1=0
 average2=0
 c1=0
 average3=0  
 for j in range(i-10,i) :
  f = f + x[j]
  average = f/10
 #print average
 for a in range(i-15,i) :
  a1 = a1 + x[a]
  average1 = a1/15
 #print average1
 for b in range(i-20,i) :
  b1 = b1 + x[b]
  average2 = b1/20
 #print average2
 for c in range(i-30,i) :
  c1 = c1 + x[c]       # Use float(x[c]) if directly using data from sheets
  average3 = c1/30
 #print average3
 input_array = input_array + [High1,Low1,Close1,Close2,Close3,Close4,Close5,average,
  average1,average2,average3]
 if count == 1 :
    d = input_array	
 else :
    d = np.vstack((d,input_array))     # creating input array of size 4873*11
'''print d.shape
print output_array.shape
print output_array[4903]
print d'''

# Building keras model

np.random.seed(7)
model=Sequential()
model.add(Dense(200,input_dim=11,activation='sigmoid'))
#model.add(LSTM(11,input_shape=(None,11),activation='sigmoid',return_sequences=True))
#model.add(LSTM(50,activation='sigmoid',return_sequences=False))
model.add(Dense(100,activation='tanh',use_bias=True))
#model.add(Dropout(0.1))
model.add(Dense(16,activation='relu',use_bias=True))
#model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))
sgd = optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='sgd',metrics=['accuracy'])

# giving input array(d) of size 4873*11 and output array of size 4873*1][2]
#output_array1 = np.array(output_array).reshape(1,4873,1)
#print output_array1.shape 
#print d1.shape
#print output_array1[0][4872]
model.fit(d,output_array,validation_split=0.33,epochs=10,batch_size=20)
test_data=np.array([d[i] for i in range(2030,2040)])
test = np.array(test_data).reshape(1,10,11)
target_data=np.array([output_array[z] for z in range(2031,2041)])
target = np.array(target_data).reshape(1,10,1)
scores,accuracy = model.evaluate(test_data,target_data)
print 'Score',scores #to train the model
#d1 = np.array(d).reshape(1,4873,11)
print 'Accuracy',accuracy
# prediction

f = []
for u in range(4873) :

#u = 4872
 z = d[u]
 sample=np.array([z])
 #print 'test sample',sample
 #print sample.shape
 f = np.append(f,model.predict(sample,verbose=1))
 
#print (d[u][2])*t.Close[index]
print f
#print (t.Close[index])*f
#print ((d[u][2])*t.Close[index])-((t.Close[index])*f)
plt.plot(f*t.Close[index],'b')
plt.plot(x*t.Close[index],'r')
plt.ylabel('Sensex_value')
plt.xlabel('Days')
g = x[30:4903]
#print g.type
z = 0
for k in range(30,4903) :
 r = (g[k]*t.Close[index])-(f[k-30]*t.Close[index])
 z = z + abs(r)
print z/4873
plt.figure()
plt.plot(g*t.Close[index]-f*t.Close[index])
plt.show()
