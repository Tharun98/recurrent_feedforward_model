
import numpy as np
import matplotlib.pyplot as plt

class neuron(object):
    def __init__(self,inputs):
        self.inputs=inputs
        self.weights=np.random.randn(inputs+1)
        self.del_w=np.zeros(inputs+1)
        self.learning_rate=0.05
    def fire_func(self,x):
        return np.tanh(x)
    def forward(self,inp):
        y=np.inner(inp,self.weights)
        out=self.fire_func(y)
        #print inp,self.weights,out
        return  out
    def backward(self,inp,output,target):
        error=target-output
        self.del_w=self.del_w+self.learning_rate*error*(inp)
        self.weights=(self.weights+self.del_w)
        return error
    def train(self,data):
        error=0
        error_array=[]
        for element in data:
            #print element[0:2]
            outp=self.forward(np.append(element[0:2],[1]))
            error=self.backward(np.append(element[0:2],[1]),outp,element[2])
            error_array.append(error)
        #print error_array
        plt.plot(error_array)
        plt.show()

class Neural_net(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes=np.zeros(input_nodes)
        self.hidden_nodes=np.zeros(hidden_nodes)
        self.output_nodes=np.zeros(output_nodes)
        self.weights_ih=(np.random.rand(input_nodes+1,hidden_nodes)-0.5)/10
        self.weights_ho=(np.random.rand(hidden_nodes+1,output_nodes)-0.5)/10
        self.delw_ih=np.zeros((input_nodes+1,hidden_nodes))
        self.delw_ho=np.zeros((hidden_nodes+1,output_nodes))
    def fire_func(self,x):
        return 1/(1+np.exp(-1.0*x))

    def forward(self,input_vector):
        self.input_nodes=input_vector
        input_vector_modified=np.append(input_vector,[1])
        self.hidden_nodes=self.fire_func(np.array([np.inner(input_vector_modified,self.weights_ih[:,i]) for i in range(len(self.hidden_nodes))]))
        hidden_vector_modified=np.append(self.hidden_nodes,[1])
        self.output_nodes=self.fire_func(np.array([np.inner(hidden_vector_modified,self.weights_ho[:,i]) for i in range(len(self.output_nodes))]))
    def backward(self,target_vector,learning_rate=0.05):
        error_o=target_vector-self.output_nodes
        diff_error_o=error_o*(1-self.output_nodes*self.output_nodes)
        diff_error_h=[(1-self.hidden_nodes*self.hidden_nodes)*([np.dot(diff_error_o,self.weights_ho[i]) for i in range(len(self.hidden_nodes))])]
        self.delw_ho=np.transpose([ learning_rate*self.weights_ho[:,i]*diff_error_o[i] for i in range(len(self.output_nodes))])
        self.weights_ho=self.weights_ho+self.delw_ho
        self.delw_ih=np.transpose([learning_rate*self.weights_ih[:,i]*diff_error_h[0][i] for i in range(len(self.hidden_nodes))])
        self.weights_ih=self.weights_ih+self.delw_ih
        return np.linalg.norm(error_o)

    def train(self,data):
        error=0
        error_array=[]
        for element in data:
            #print element[0:2]
            self.forward(element)
            error=self.backward(element)
            error_array.append(error)
        #print error_array
        plt.plot(error_array)
        plt.show()


network=Neural_net(8,3,8)

data=[1,0,0,0,0,0,0,0]
network.forward(data)
print network.input_nodes
print network.hidden_nodes
print network.output_nodes
'''
print network.weights_ho
print network.weights_ih
network.backward(data)
print network.weights_ih
print network.weights_ho
'''

'''
inputs_data=[[0 if r[0]<0 else 1,0 if r[1]<0 else 1] for r in np.random.randn(200,2) ]
[r.append(r[0]|r[1]) for r in inputs_data]
neu=neuron(2)
neu.train(inputs_data)
'''


data_samples=np.identity(8)
input_data=[data_samples[np.random.randint(0,8)] for i in range(2000)]
network.train(input_data)
test_sample=data_samples[np.random.randint(0,8)]
network.forward(test_sample)
print 'test sample= ',test_sample
print 'hidden ',network.hidden_nodes
print 'output = ',network.output_nodes

'''
input_data=[[1,0,0,0,0,0,0,0] for i in range(5000)]
network.train(input_data)
'''
'''
print 'test sample= ',test_sample
print 'hidden ',network.hidden_nodes
print 'output = ',network.output_nodes
'''
