import Models.Izhikevich.Neuron as neu
import Models.Izhikevich.Connection as con
import matplotlib.pyplot as plt




class NeuralNetwork:


    def __init__(self, name):
        self.name = name
        self.neurons = {}
        self.connections = []

    def __str__(self):
        return """ 
               [
               %s,
               %s,
               %s
               ]
                """ % (self.name, " ".join(str(c) for c in self.neurons.values())," ".join(str(c) for c in self.connections))
        # return "[neu: %s, estado: %s, conexiones: %s]" % (self.neuronName,self.neuronState,len(self.neuronConnections))

    def __repr__(self):
        return "[nombre: %s,  neuronas: %s,conexiones: %s]" % (
        self.name, self.countNeurons(), self.countConnections)

    def updateStateToTrace(self):
        for neu in self.neurons.values():
            neu.updateStateToTrace()
        for con in self.connections:
            con.updateStateToTrace()
    def graphVariableTraces(self,folder):
        for neu in self.neurons.values():
            neu.graphVariableTraces(folder)
        for con in self.connections:
            con.graphVariableTraces(folder)
    def getNeuron(self, nname):
         return self.neurons[nname]
    def getNeurons(self):
         return self.neurons.values()
    def getNeuronNames(self):
         return self.neurons.keys()

    def resetAllNeurons(self):
        for n in self.neurons.values():
            n.resetPotencial()

    def countNeurons(self):
        return len(self.neurons)


    def countConnections(self, type=None):
        count = 0
        if type==None:
            count=len(self.connections)
        else:
            for conn in self.connections:
                if conn.isType(type):
                    count = count + 1
        return count

    def doSimulationStep(self,delta):
        for neu in self.neurons.values():
            neu.computeV(delta,self.getDendriticConnectionsFor(neu))



    def getDendriticConnectionsFor(self,targetNeuron):
        dendriticConnections = [c for c in self.connections if c.getTarget() == targetNeuron]
        return dendriticConnections

    def getConnectionIdx(self,idx):
        return self.connections[idx]
    def getConnections(self):
        return self.connections


    def getConnectionTestWeightOfIdx(self,idx):
        return self.connections[idx].getTestWeight()

    def setConnectionTestWeightOfIdx(self,idx,val):
        return self.connections[idx].setTestWeight(val)

    def getConnectionTestSigmaOfIdx(self,idx):
        return self.connections[idx].getTestSigma()

    def setConnectionTestSigmaOfIdx(self,idx,val):
        return self.connections[idx].setTestSigma(val)

    def getNeuronPotencialOfName(self,name):
        return self.neurons[name].getPotencial()



    def getNeuronTestParamAOfName(self,name):
        return self.neurons[name].getTestParamA()
    def setNeuronTestParamAOfName(self,name,val):
        return self.neurons[name].setTestParamA(val)
    def getNeuronTestParamBOfName(self,name):
        return self.neurons[name].getTestParamB()
    def setNeuronTestParamBOfName(self,name,val):
        return self.neurons[name].setTestParamB(val)
    def getNeuronTestParamCOfName(self,name):
        return self.neurons[name].getTestParamC()
    def setNeuronTestParamCOfName(self,name,val):
        return self.neurons[name].setTestParamC(val)
    def getNeuronTestParamDOfName(self,name):
        return self.neurons[name].getTestParamD()
    def setNeuronTestParamDOfName(self,name,val):
        return self.neurons[name].setTestParamD(val)

    def getConnectionSize(self):
        return len(self.connections)

    def getNeuronsSize(self):
        return len(self.neurons)

    def resetActivationTraces(self):
        for neu in self.neurons.values():
            neu.resetLastActivationTrace()

    def recordActivationTraces(self):
        for neu in self.neurons.values():
            neu.recordActivationTrace()



    def writeToFile(self,logfile):
        logfile.write("NEURONS"+'\n')
        for neu in self.neurons.values():
            logfile.write(str(neu) + '\n')
        logfile.write("CONNECTIONS"+'\n')
        for con in self.connections:
            logfile.write(str(con) + '\n')

    def writeToGraphs(self,folder):
        times=[]
        values=[]
        for neu in self.neurons.values():
            values=neu.getLastActivaionTrace()
            neu.resetLastActivationTrace()
            times=[i for i in range(0, len(values))]
            fig=plt.figure()
            plt.plot(times, values, 'ro', label='voltage')
            plt.ylabel('Voltages')
            plt.xlabel('Episode step')
            plt.legend()
            plt.title('Neural Voltage')
            fig.savefig( folder+'/'+neu.getName()+'.png', bbox_inches='tight')



    #needed for PyGAD
    def getIndividualForPYGAD(self):
        currList = []
        for neu in self.neurons.values():
            for nc in neu.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        for con in self.connections:
            for nc in con.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        return currList

    #needed for HyperOpt
    def getSpaceForHyperOpt(self):
        currDic={}
        varDic = {}
        #commented to learn only connections
        for neu in self.neurons.values():
            varDic=neu.getVariablesForHyperOpt()
            for key,value in varDic.items():
                currDic[key]=value
        for index in range(0,len(self.connections)):
            con=self.connections[index]
            varDic=con.getSpaceForHyperOpt(index)
            for key,value in varDic.items():
                currDic[key]=value
        return currDic

    def commitNoise(self):
        #commented if only learn on connections
        for neu in self.neurons.values():
            neu.commitNoise()
        for con in self.connections:
            con.commitNoise()

    def revertNoise(self):
        #commented if only learn on connections
        for neu in self.neurons.values():
            neu.revertNoise()
        for con in self.connections:
            con.revertNoise()

    def isSameAs(self,neuralNetwork2):
        sameNeuronsList=[]
        sameConnectionsList=[]
        for key in self.neurons:
            same=(self.neurons[key]).isSameAs(neuralNetwork2.neurons[key])
            if not same:
                sameNeuronsList.append(self.neurons[key])
        for index  in range(0,len(self.connections)):
            same=(self.connections[index]).isSameAs(neuralNetwork2.connections[index])
            if not same:
                sameConnectionsList.append(self.connections[index])
        sameNeuronsList.append(sameConnectionsList)
        return sameNeuronsList



def loadNeuralNetwork(xmlNn):
     nnToReturn = NeuralNetwork(xmlNn.attrib['name'])
     for child in xmlNn:
         if child.tag == 'neuron':
             n = neu.loadNeuron(child)
             nnToReturn.neurons[n.getName()] = n
         if child.tag == 'connection':
              c = con.loadConnection(child,nnToReturn)
              nnToReturn.connections.append(c)
     return nnToReturn


