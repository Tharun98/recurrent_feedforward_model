
class Neural_net(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        #vectors to store nuron outputs
        self.input_nodes=np.zeros(input_nodes)
        self.hidden_nodes=np.zeros(hidden_nodes)
        self.output_nodes=np.zeros(output_nodes)
        #weigh matrix in which each row represents weights arrising from one nuron and each colomn represents weights reaching nuron of next layer ,extra weight is present for bias
        self.weights_ih=(np.random.rand(input_nodes+1,hidden_nodes)-0.5)/10     #input ,hidden matrix
        self.weights_ho=(np.random.rand(hidden_nodes+1,output_nodes)-0.5)/10    #hidden , output matrix
        #matrix to store chage in weights
        self.delw_ih=np.zeros((input_nodes+1,hidden_nodes))
        self.delw_ho=np.zeros((hidden_nodes+1,output_nodes))
    def fire_func(self,x):
        return 1/(1+np.exp(-1.0*x))         #nuron firing

    def forward(self,input_vector):         #function that feeds an input to network
        self.input_nodes=input_vector
        #modifed to allow the contribution of bias weght
        input_vector_modified=np.append(input_vector,[1])
        #input vector is multiplied with weights reaching each nuron of next layer to get next nuron output
        self.hidden_nodes=self.fire_func(np.array([np.dot(input_vector_modified,self.weights_ih[:,i]) for i in range(len(self.hidden_nodes))]))
        hidden_vector_modified=np.append(self.hidden_nodes,[1])
        self.output_nodes=self.fire_func(np.array([np.dot(hidden_vector_modified,self.weights_ho[:,i]) for i in range(len(self.output_nodes))]))
    def backward(self,target_vector,learning_rate=0.05):        #function for updating weights using back propagation
        error_o=target_vector-self.output_nodes
        diff_error_o=error_o*(1-self.output_nodes*self.output_nodes)  #error term calculation of output layer
        diff_error_h=[(1-self.hidden_nodes*self.hidden_nodes)*([np.dot(diff_error_o,self.weights_ho[i]) for i in range(len(self.hidden_nodes))])]       #error term derived from error term of previous layer
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

data_samples=np.identity(8)
input_data=[data_samples[np.random.randint(0,8)] for i in range(2000)]
network.train(input_data)
test_sample=data_samples[np.random.randint(0,8)]
network.forward(test_sample)
print 'test sample= ',test_sample
print 'hidden ',network.hidden_nodes
print 'output = ',network.output_nodes
