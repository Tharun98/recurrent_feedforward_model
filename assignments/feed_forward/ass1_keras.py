import numpy as np
from keras.models import Sequential,optimizers
from keras.layers import Dense,Dropout,Activation,Flatten
np.random.seed(7)
def normalize(data):
    return data*0.8+0.1
def denoramalize(out_data):
    return (out_data-0.1)/0.8
'''
data_samples=np.identity(8)
data=np.array([data_samples[np.random.randint(0,8)] for i in range(10)])
print 'data ',data
norm_data=normalize(data)
print 'normalized data ',norm_data
denorm_data=denoramalize(norm_data)
print 'denormalilzed data ',denorm_data
'''


model=Sequential()
model.add(Dense(9,input_dim=8,activation='sigmoid'))
model.add(Dense(3,activation='sigmoid'))
model.add(Dense(8,activation='sigmoid'))
sgd = optimizers.SGD(lr=0.2, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.compile(loss='binary_crossentropy',optimizer='adam')
data_samples=np.identity(8)
data=np.array([data_samples[np.random.randint(0,8)] for i in range(10000)])
input_data=normalize(data)
model.fit(input_data,input_data,epochs=16,batch_size=10)
test_data=np.array([data_samples[np.random.randint(0,8)] for i in range(10)])
test_data_norm=normalize(test_data)
scores=model.evaluate(test_data_norm,test_data_norm)
print scores

sample=np.array([data_samples[np.random.randint(0,8)] ])
sample_norm=normalize(sample)
print 'test sample',sample
output=model.predict(sample,verbose=1)
received_output=denoramalize(output)
print received_output

#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
