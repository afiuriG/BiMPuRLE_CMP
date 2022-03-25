import Models.Izhikevich.NeuralNetwork as nN
import Models.Izhikevich.Connection as con
import Models.Izhikevich.ModelInterfaces as mi
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.cm as cm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from Utils import GraphUtils as gut
import pickle

np.random.seed(423)
neuronRandomIndexedNames = []

class Model:
    def __init__(self,name):
        self.name = name
        self.neuralnetwork = None
        self.interfaces = {}
        self.generationGraphModeFolder=''
        self.generationGraphModeCounter=0

    def setRandomIndexesList(self,list):
        global neuronRandomIndexedNames
        neuronRandomIndexedNames=list


    def setGenerationGraphModeFolderOn(self,folder):
        self.generationGraphModeFolder=folder
        #same ammount of steps so the generation graph is self reseting
        self.generationGraphModeCounter=5

    def setGenerationGraphModeFolderOff(self):
        self.generationGraphModeFolder= ''



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




    def getInterface(self,name):
        return self.interfaces[name]

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




# The name was taken from the original paper (for I&F) but I think some thing like runConnectome
# should be more appropriated due to takes observations as input, run the connectome with
# delta as iteration step (to solve the EDO) and so can be read the result in the proper output neurons.
    def Update(self,observations,mode=None,doLog=False):
        deltaT=0.1
        simmulationSteps = 5
        self.interfaces['OUT1'].resetFired()
        if doLog:
            networkLog = open('log/oneEpisodeModel.log', 'a')
            networkLog.write("----Update(%s,%s,%s)" % (observations,deltaT,simmulationSteps) + '\n')
            #networkLog.write('\t' + "Pre state" + '\n')
        if self.generationGraphModeCounter != 0:
             gut.graphSnapshot('IZ',self, 0, True)
        for step in range(0,simmulationSteps):
            #put input values
            self.interfaces['IN1'].setValue(observations[0])
            self.interfaces['IN2'].setValue(observations[1])
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            self.neuralnetwork.doSimulationStep(deltaT)
            # recover the output values
            ret = self.interfaces['OUT1'].getFeedBackNN()
            if self.generationGraphModeCounter!=0:
                gut.graphSnapshot('IZ',self, step+1, True)
                self.generationGraphModeCounter=self.generationGraphModeCounter-1
            if doLog:
                #self.neuralnetwork.writeToFile(networkLog)
                networkLog.write('\t' + "Return:%s"%(ret) + '\n')
                networkLog.write('\t' + "------------------------------------" + '\n')
                networkLog.flush()
        if mode=='debug':
            self.neuralnetwork.recordActivationTraces()
        # return the value of the output
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


    def getModelInterface(self):
        return mi

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
        #same as commitNoise
        self.neuralnetwork.revertNoise()

    #########################&&&&&&&&&&&&&&&&&&&&&&&&
    def addNoise(self,parameter,varianza,samplesAmmount):
        mu, sigma = 0, varianza
        if(parameter == 'a'):
            #normalDistribuitedValues = np.random.normal(0.51, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(0.019, 0.021, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
            #print('inside addNoise %s for a: %s' % (samplesAmmount,uniformDistribuitedValues))
        elif parameter == 'b':
            #normalDistribuitedValues = np.random.normal(0.225, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(0.19, 0.21, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
        elif parameter == 'c':
            #normalDistribuitedValues = np.random.normal(-60.0, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(-21.0, -20.0, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
        elif parameter == 'd':
            #normalDistribuitedValues = np.random.normal(5, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(7, 8, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
        elif parameter == 'Sigma':
            #normalDistribuitedValues = np.random.normal(0.25, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(0.0, 1, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
        elif parameter == 'Weight':
            #normalDistribuitedValues = np.random.normal(2, sigma, samplesAmmount)
            normalDistribuitedValues = np.random.uniform(0, 5, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
            #print('inside addNoise %s for weight: %s' % (samplesAmmount, uniformDistribuitedValues))
        self.noiseParam(samplesAmmount,parameter,uniformDistribuitedValues,normalDistribuitedValues)


    def noiseParam(self, samplesAmmount, parameter, whiches, values):
        for i in range(0, samplesAmmount):
            which = whiches[i]
            x = values[i]
            if parameter == 'Weight':
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                #newValue = self.neuralnetwork.getConnectionTestWeightOfIdx(which) + x
                newValue =  x
                if newValue < 0:
                    newValue = 0
                elif newValue > 5.0:
                    newValue = 5.0
                self.neuralnetwork.setConnectionTestWeightOfIdx(which, newValue)
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'Sigma':
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                #newValue = self.neuralnetwork.getConnectionTestSigmaOfIdx(which) + x
                newValue = x
                if newValue < 0:
                    newValue = 0
                elif newValue > 1.0:
                    newValue = 1.0
                self.neuralnetwork.setConnectionTestSigmaOfIdx(which, newValue)
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'a':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestParamAOfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < 0.019999:
                    newValue = 0.019999
                elif newValue > 0.020001:
                    newValue = 0.020001
                self.neuralnetwork.setNeuronTestParamAOfName(neuronRandomIndexedNames[which],newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'b':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestParamBOfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < 0.19999:
                    newValue = 0.19999
                elif newValue > 0.20001:
                    newValue = 0.20001
                self.neuralnetwork.setNeuronTestParamBOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'c':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestParamCOfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < -21.0:
                    newValue = -21.0
                elif newValue > -20.0:
                    newValue = -20.0
                self.neuralnetwork.setNeuronTestParamCOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron indxe %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'd':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestParamDOfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < 7.0:
                    newValue = 7.0
                elif newValue > 8.0:
                    newValue = 8.0
                self.neuralnetwork.setNeuronTestParamDOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))







