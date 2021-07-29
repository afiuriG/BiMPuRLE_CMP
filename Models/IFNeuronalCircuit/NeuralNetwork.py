import Models.IFNeuronalCircuit.Neuron as neu
import Models.IFNeuronalCircuit.Connection as con

V_Leak = -70
Sigmoid_sigma = 0.1
G_Leak = 1
Cm = 0.05


class NeuralNetwork:


    def __init__(self, name):
        self.name = name
        self.neurons = {}
        self.connections = []

    def __str__(self):
        return "[nombre: %s, neuronas: %s, conexiones: %s]" % (
        self.name, " ".join(str(c) for c in self.neurons.values())," ".join(str(c) for c in self.connections))
        # return "[neu: %s, estado: %s, conexiones: %s]" % (self.neuronName,self.neuronState,len(self.neuronConnections))

    def __repr__(self):
        return "[nombre: %s,  neuronas: %s,conexiones: %s]" % (
        self.name, self.countNeurons(), self.countConnections)



    def loadNeuron(self, nname):
        if nname not in self.neurons:
             n = neu.Neuron(nname)
             n.initialize(G_Leak, V_Leak, Cm)
             self.neurons[nname] = n

    def getNeuron(self, nname):
         return self.neurons[nname]
    def getNeurons(self):
         return self.neurons.values()
    def getNeuronNames(self):
         return self.neurons.keys()

    def resetAllNeurons(self):
        for n in self.neurons.values():
            n.resetPotencial(V_Leak)

    def countNeurons(self):
        return len(self.neurons)

    def loadConnection(self, conType, sourceName, targetName, weight):
        srcNeu = self.neurons[sourceName]
        tarNeu = self.neurons[targetName]
        newCon = con.Connection(conType, srcNeu, tarNeu, weight, Sigmoid_sigma)
        self.connections.append(newCon)

    def countConnections(self, type=None):
        count = 0
        if type==None:
            count=len(self.connections)
        else:
            for conn in self.connections:
                if conn.isType(type):
                    count = count + 1
        return count

    def doSimulationStep(self, delta):
        for neu in self.neurons.values():
            neu.computeVnext(delta, self.getDendriticConnectionsFor(neu))
            #neu.useVnext()?????? me parece que este explota



    def getDendriticConnectionsFor(self,targetNeuron):
        dendriticConnections = [c for c in self.connections if c.getTarget() == targetNeuron]
        return dendriticConnections

    def getConnectionIdx(self,idx):
        return self.connections[idx]

    def getConnectionTestWeightOfIdx(self,idx):
        return self.connections[idx].getTestWeight()

    def setConnectionTestWeightOfIdx(self,idx,val):
        return self.connections[idx].setTestWeight(val)

    def getConnectionTestSigmaOfIdx(self,idx):
        return self.connections[idx].getTestSigma()

    def setConnectionTestSigmaOfIdx(self,idx,val):
        return self.connections[idx].setTestSigma(val)


    def getNeuronTestVleakOfName(self,name):
        return self.neurons[name].getTestVleak()
    def setNeuronTestVleakOfName(self,name,val):
        return self.neurons[name].setTestVleak(val)

    def getNeuronTestGleakOfName(self,name):
        return self.neurons[name].getTestGleak()
    def setNeuronTestGleakOfName(self,name,val):
        return self.neurons[name].setTestGleak(val)


    def getNeuronTestCmOfName(self,name):
        return self.neurons[name].getTestCm()
    def setNeuronTestCmOfName(self,name,val):
        return self.neurons[name].setTestCm(val)


    def getState(self):
        st='-----------\n'
        for neu in self.neurons.values():
            st=st+str(neu)+'\n'
        for con in self.connections:
            st=st+str(con)
        return st

    def getConnectionSize(self):
        return len(self.connections)

    def getNeuronsSize(self):
        return len(self.neurons)


    def commitNoise(self):
        for neu in self.neurons.values():
            neu.commitNoise()
        for con in self.connections:
            con.commitNoise()

    def revertNoise(self):
        for neu in self.neurons.values():
            neu.revertNoise()
        for con in self.connections:
            con.revertNoise()


    def writeToFile(self,logfile):
        logfile.write("NEURONS"+'\n')
        for neu in self.neurons.values():
            logfile.write(str(neu) + '\n')
        logfile.write("CONNECTIONS"+'\n')
        for con in self.connections:
            logfile.write(str(con) + '\n')

    def dumpNeuralNetwork(self,xmlNn):
        xmlNn.set('name',self.name)
        #xmlNeurons = ET.SubElement(xmlNn, 'Neurons')
        #xmlConnections = ET.SubElement(xmlNn, 'Connections')
        for neu in self.neurons.values():
            neu.dumpNeuron(xmlNn)
        for con in self.connections:
            con.dumpConnection(xmlNn)

    def getIndividualForPYGAD(self):
        currList = []
        for neu in self.neurons.values():
            for nc in neu.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        for con in self.connections:
            for nc in con.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        return currList

    def getSpaceForHyperOpt(self):
        currDic={}
        varDic = {}
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



    def isSameAs(self,neuralNetwork2):
        sameNeurons=True
        sameConnections=True
        for key in self.neurons:
            sameNeurons=sameNeurons or (self.neurons[key]).isSameAs(neuralNetwork2.neurons[key])
        for index  in range(0,len(self.connections)):
            sameConnections = sameConnections or (self.connections[index]).isSameAs(neuralNetwork2.connections[index])
        return [sameNeurons,sameConnections]

    def clone(self):
        nnToRet=NeuralNetwork(self.name)
        for key,val in self.neurons.items():
            nnToRet.neurons[key]=(self.neurons[key]).clone()
        for con in self.connections:
            nnToRet.connections.append(con.clone())
        return nnToRet


def loadNeuralNetwork(xmlNn):
     nnToReturn = NeuralNetwork(xmlNn.attrib['name'])
     #namesForRandomIndexes = []
     for child in xmlNn:
         if child.tag == 'neuron':
             n = neu.loadNeuron(child)
             nnToReturn.neurons[n.getName()] = n
             #namesForRandomIndexes.append(n.getName())
         if child.tag == 'connection':
              c = con.loadConnection(child,nnToReturn)
              nnToReturn.connections.append(c)
     return nnToReturn


