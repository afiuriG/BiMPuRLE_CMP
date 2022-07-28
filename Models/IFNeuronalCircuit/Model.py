import Models.IFNeuronalCircuit.NeuralNetwork as nN
import Models.IFNeuronalCircuit.ModelInterfaces as mi
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from Utils import GraphUtils as gut
from Utils import Environment as eut

np.random.seed(423)
neuronRandomIndexedNames = []



class Model:
    def __init__(self,name):
        self.name = name
        self.neuralnetwork = None
        self.interfaces = {}
        self.generationGraphModeFolder = ''
        self.generationGraphModeCounter=0

    def updateStateToTrace(self):
        return self.neuralnetwork.updateStateToTrace()

    def graphVariableTraces(self,folder):
        self.neuralnetwork.graphVariableTraces(folder)


    def setRandomIndexesList(self,list):
        global neuronRandomIndexedNames
        neuronRandomIndexedNames=list

    def load(self,fromFile=''):
        if fromFile=='':
           print('the harcoded loading functionality was removed.')
        else:
           self.loadFromFile(fromFile)

    def __str__(self):
        return """
        RED:
        %s
        INTERFACES:
        %s
        """ % (self.neuralnetwork," ".join(str(i) for i in self.interfaces.values()))
    def __repr__(self):
        return """RED:
            %s
        INTERFACES:
            %s
            """ % (self.neuralnetwork, " ".join(str(i) for i in self.interfaces))


    def setGenerationGraphModeFolderOn(self,folder):
        self.generationGraphModeFolder=folder
        #same ammount of steps so the generation graph is self reseting
        self.generationGraphModeCounter=10

    def setGenerationGraphModeFolderOff(self):
        self.generationGraphModeFolder= ''

    def getInterface(self,name):
        return self.interfaces[name]
    def getInterfaces(self):
        return self.interfaces.values()


    def getNeuron(self,name):
        return self.neuralnetwork.getNeuron(name)

    def getName(self):
        return self.name
    def setName(self,name):
        self.name = name

#Reset
    def Reset(self):
        self.neuralnetwork.resetAllNeurons()
        for intf in self.interfaces.values():
            intf.reset()

    def runPulse(self):
        deltaT = 0.1
        simmulationSteps = 10
        for step in range(0, simmulationSteps):
            randomX, randomV = eut.getRandomObservation("MouCarCon")
            self.interfaces['IN1'].setValue(randomX)
            self.interfaces['IN2'].setValue(randomV)
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            self.neuralnetwork.doSimulationStep(deltaT)
            ret = self.interfaces['OUT1'].getFeedBackNN()
        #print("x:%s, v:%s, a:%s"%(randomX,randomV,ret))



# The name was taken from the original paper but I think some thing like runConnectome
# should be more appropriated due to takes observations as input, run the connectome with
# delta as iteration step (to solve the EDO) and so can be read the result in the proper output neurons.
    def Update(self,observations,mode=None,doLog=False):
        deltaT=0.1
        simmulationSteps = 10
        if doLog:
            networkLog = open('log/oneEpisodeModel.log', 'a')
            networkLog.write("----Update(%s,%s,%s)" % (observations,deltaT,simmulationSteps) + '\n')
        if self.generationGraphModeCounter != 0:
             gut.graphSnapshot('IF',self, 0, True)
        for step in range(0,simmulationSteps-1):
            #put input values
            self.interfaces['IN1'].setValue(observations[0])
            self.interfaces['IN2'].setValue(observations[1])
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            self.neuralnetwork.doSimulationStep(deltaT)
            # recover the output
            ret = self.interfaces['OUT1'].getFeedBackNN()
            if self.generationGraphModeCounter!=0:
                gut.graphSnapshot('IF',self,step+1,True)
                self.generationGraphModeCounter=self.generationGraphModeCounter-1
            if doLog:
                networkLog.write('\t' + "Return:%s"%(ret) + '\n')
                networkLog.write('\t' + "------------------------------------" + '\n')
                networkLog.flush()
        if mode == 'debug':
            self.neuralnetwork.recordActivationTraces()
        #return the value of the output
        return ret



    def writeToFile(self,logfile):
        logfile.write("NEURAL NETWORK" + '\n')
        self.neuralnetwork.writeToFile(logfile)
        logfile.write("INTERFACES" + '\n')
        for inter in self.interfaces.values():
            logfile.write(str(inter) + '\n')


    def isSameAs(self,model2):
        #should return an empty list if are the same but not empty in other case
        sameNetList= self.neuralnetwork.isSameAs(model2.neuralnetwork)
        sameIntList=[]
        for inter in self.interfaces.values():
            same=inter.isSameAs(model2.getInterface(inter.getName()))
            if not same :
                sameIntList.append(inter)
        sameNetList.append(sameIntList)
        return sameNetList


    def loadFromFile(self,fromFile):
        mydoc = ET.parse(fromFile)
        xmlRootModelElement = mydoc.getroot()
        for child in xmlRootModelElement:
            if child.tag == 'NeuralNetwork':
               loaded =  nN.loadNeuralNetwork(child)
               self.neuralnetwork = loaded
               for n in self.neuralnetwork.getNeurons():
                   neuronRandomIndexedNames.append(n.getName())
            if child.tag == 'Interfaces':
               inter = mi.loadConnection(child,self.neuralnetwork)
               self.interfaces=inter


    def getModelInterfaces(self):
        return self.interfaces

    def getNeuronRandomIndexedNames(self):
        global neuronRandomIndexedNames
        return neuronRandomIndexedNames


#needed for RandomSeek

    # CommitNoise()
    def commitNoise(self):
        #the commit is on all the neurons, may be could be more efficient if only are commited the updated ones.
        self.neuralnetwork.commitNoise()

    # UndoNoise
    def revertNoise(self):
        # same as commitNoise
        self.neuralnetwork.revertNoise()


    def addNoise(self,parameter,varianza,samplesAmmount):
        mu, sigma = 0, varianza
        if (parameter == 'Gleak'):
            #normalDistribuitedValues = np.random.uniform(0.05, 5, samplesAmmount)
            normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
            # print('sigma:%s'%(sigma))
            # print(normalDistribuitedValues)
        elif parameter == 'Vleak':
            #normalDistribuitedValues = np.random.uniform(-90, 0, samplesAmmount)
            normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
            #print('inside the addNoise %s for vleak: %s' % (samplesAmmount,uniformDistribuitedValues))
        elif parameter == 'Cm':
            #normalDistribuitedValues = np.random.uniform(0.001, 1.0, samplesAmmount)
            normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
        elif parameter == 'Sigma':
            #normalDistribuitedValues = np.random.uniform(0.05, 0.5, samplesAmmount)
            normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
        elif parameter == 'Weight':
            #normalDistribuitedValues = np.random.uniform(0.0, 3.0, samplesAmmount)
            normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
            #print('inside the addNoise %s for weight: %s'%(samplesAmmount,uniformDistribuitedValues))
        self.noiseParam(samplesAmmount,parameter,uniformDistribuitedValues,normalDistribuitedValues)


    def noiseParam(self, samplesAmmount, parameter, whiches, values):
        for i in range(0, samplesAmmount):
            which = whiches[i]
            x = values[i]
            if parameter == 'Weight':
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                newValue = self.neuralnetwork.getConnectionTestWeightOfIdx(which) + x
                #newValue=x
                if newValue < 0:
                    newValue = 0
                elif newValue > 3.0:
                    newValue = 3.0
                self.neuralnetwork.setConnectionTestWeightOfIdx(which, newValue)
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'Sigma':
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                newValue = self.neuralnetwork.getConnectionTestSigmaOfIdx(which) + x
                #newValue = x
                if newValue < 0.05:
                    newValue = 0.05
                elif newValue > 0.5:
                    newValue = 0.5
                self.neuralnetwork.setConnectionTestSigmaOfIdx(which, newValue)
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'Gleak':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestGleakOfName(neuronRandomIndexedNames[which]) + x
                #newValue = x
                if newValue < 0.05:
                    newValue = 0.05
                elif newValue > 5.0:
                    newValue = 5.0
                self.neuralnetwork.setNeuronTestGleakOfName(neuronRandomIndexedNames[which],newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'Vleak':
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestVleakOfName(neuronRandomIndexedNames[which]) + x
                #newValue = x
                if newValue < -90:
                    newValue = -90
                elif newValue > 0:
                    newValue = 0
                self.neuralnetwork.setNeuronTestVleakOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'Cm':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestCmOfName(neuronRandomIndexedNames[which]) + x
                #newValue = x
                if newValue < 0.001:
                    newValue = 0.001
                elif newValue > 1.0:
                    newValue = 1.0
                self.neuralnetwork.setNeuronTestCmOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))






