
import numpy as np
import matplotlib.pyplot as plt

class neuron(object):
    def __init__(self,inputs):
        #We initiate the neuron with learinig rate and weights
        self.inputs=inputs
        self.weights=np.random.randn(inputs+1)
        self.del_w=np.zeros(inputs+1)
        self.learning_rate=0.05
    #activation funciton for the neuron
    #in this case we use hyperbolic tangent
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
        #vectors to store nuron outputs
        self.input_nodes=np.zeros(input_nodes)
        self.hidden_nodes=np.zeros(hidden_nodes)
        self.output_nodes=np.zeros(output_nodes)
        #weigh matrix in which each row represents weights arrising from one nuron and each colomn represents weights reaching nuron of next layer ,extra weight is present for bias
        self.weights_ih=(np.random.rand(input_nodes+1,hidden_nodes)-0.5)/10     #input ,hidden matrix
        self.weights_ho=(np.random.rand(hidden_nodes,output_nodes)-0.5)/10    #hidden , output matrix
        #matrix to store chage in weights
        self.delw_ih=np.zeros((input_nodes+1,hidden_nodes))
        self.delw_ho=np.zeros((hidden_nodes,output_nodes))
    def fire_func(self,x):
        return np.tanh(x)        #nuron firing

    def forward(self,input_vector):         #function that feeds an input to network
        self.input_nodes=input_vector
        #modifed to allow the contribution of bias weight
        input_vector_modified=np.append(input_vector,[1])
        #input vector is multiplied with weights reaching each nuron of next layer to get next nuron output
        self.hidden_nodes=self.fire_func(np.array([np.dot(input_vector_modified,self.weights_ih[:,i]) for i in range(len(self.hidden_nodes))]))
        hidden_vector_modified=self.hidden_nodes
        self.output_nodes=self.fire_func(np.array([np.dot(hidden_vector_modified,self.weights_ho[:,i]) for i in range(len(self.output_nodes))]))
    def backward(self,target_vector,learning_rate=0.0000001):        #function for updating weights using back propagation
        error_o=target_vector-self.output_nodes
        diff_error_o=error_o*(1-self.output_nodes*self.output_nodes)  #error term calculation of output layer
        diff_error_h=[(1-self.hidden_nodes*self.hidden_nodes)*([np.dot(diff_error_o,self.weights_ho[i]) for i in range(len(self.hidden_nodes))])]       #error term derived from error term of previous layer
        #print diff_error_h
        self.delw_ho=np.transpose([ learning_rate*self.weights_ho[:,i]*diff_error_o[i] for i in range(len(self.output_nodes))])             #change in weights
        self.weights_ho=self.weights_ho+self.delw_ho                #updating weights
        self.delw_ih=np.transpose([learning_rate*self.weights_ih[:,i]*diff_error_h[0][i] for i in range(len(self.hidden_nodes))])           #change in weights
        self.weights_ih=self.weights_ih+self.delw_ih                #updating weights
        return np.linalg.norm(error_o)                  #error term

    def train(self,data):           #training network
        error=0
        error_array=[]
        for element in data:
            #print element[0:2]
            self.forward(element)
            error=self.backward(element)
            error_array.append(error)
        #print error_array
        plt.plot(error_array)           #plotting error
        plt.show()


network=Neural_net(8,3,8)
'''
data=[1,0,0,0,0,0,0,0]
network.forward(data)
print 'input nodes',network.input_nodes
print 'hidden nodes',network.hidden_nodes
print 'output nodes',network.output_nodes
print 'ih weights',network.weights_ih
print 'ho weights',network.weights_ho
network.backward(data)
print 'weights ch ho',network.delw_ho
print 'weghts ch ih',network.delw_ih
print 'ih weights',network.weights_ih
print 'ho weights',network.weights_ho
'''
'''
print network.weights_ho
print network.weights_ih
network.backward(data)
print network.weights_ih
print network.weights_ho
'''

'''
inputs_data=[[0 if r[0]<0 else 1,0 if r[1]<0 else 1] for r in np.random.randn(200,2) ]
input_data=[r.append(r[0]|r[1]) for r in inputs_data]
neu=neuron(2)
neu.train(inputs_data)
'''


data_samples=np.identity(8)
input_data=[data_samples[np.random.randint(0,8)] for i in range(10000)]
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
