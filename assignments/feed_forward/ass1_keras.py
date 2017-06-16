import numpy as np
from keras.models import Sequential,optimizers
from keras.layers import Dense,Dropout,Activation,Flatten
np.random.seed(7)
model=Sequential()
model.add(Dense(9,input_dim=8,activation='sigmoid'))
model.add(Dense(3,activation='sigmoid'))
model.add(Dense(8,activation='sigmoid'))
sgd = optimizers.SGD(lr=0.2, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.compile(loss='binary_crossentropy',optimizer='adam')
data_samples=np.identity(8)
input_data=np.array([data_samples[np.random.randint(0,8)] for i in range(100000)])
model.fit(input_data,input_data,epochs=5,batch_size=10)
test_data=np.array([data_samples[np.random.randint(0,8)] for i in range(10)])
scores=model.evaluate(test_data,test_data)
print scores

sample=np.array([data_samples[np.random.randint(0,8)] ])
print 'test sample',sample
print(model.predict(sample,verbose=1))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
