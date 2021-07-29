import xml.etree.ElementTree as ET
import numpy as np

class BinaryInterface:

    def __init__(self, name, posNeu, negNeu, type):
        self.name = name
        self.maxPotencial = -20
        self.minPotencial = -70
        self.maxValue = 1
        self.minValue = -1
        self.value = 0
        self.positiveNeuron = posNeu
        self.negativeNeuron = negNeu
        self.type = type

    def __str__(self):
        return "<nombre: %s, tipo: %s, pos: %s, neg: %s, value: %s>" % (
            self.name, self.type, self.positiveNeuron, self.negativeNeuron, self.value)

    def __repr__(self):
        return "[pos: %s, neg: %s, value: %s]" % (
            self.positiveNeuron, self.negativeNeuron, self.value)

    def reset(self):
        self.value = 0

    def getName(self):
        return  self.name

    def setMaxValue(self,max):
        self.maxValue = max

    def setMinValue(self,min):
        self.minValue = min


    def setValue(self, val):
        self.value = val

    #llamado sync en el codigo del autor
    def feedNN(self):
        if self.type == 'IN':
            if self.value>=0:
                corVal = self.value/self.maxValue
                pot = (self.maxPotencial-self.minPotencial)*corVal+self.minPotencial
                self.positiveNeuron.setPotencial(pot)
                self.positiveNeuron.setTestPotencial(pot)
                self.negativeNeuron.setPotencial(self.minPotencial)
                self.negativeNeuron.setTestPotencial(self.minPotencial)
            else:
                corVal = self.value/(-self.minValue)
                pot = (self.maxPotencial-self.minPotencial)*(-corVal)+self.minPotencial
                self.negativeNeuron.setPotencial(pot)
                self.negativeNeuron.setTestPotencial(pot)
                self.positiveNeuron.setPotencial(self.minPotencial)
                self.positiveNeuron.setTestPotencial(self.minPotencial)

    def getFeedBackNN(self):
        if self.type == 'OUT':
            negPot = self.negativeNeuron.getPotencial()
            posPot = self.positiveNeuron.getPotencial()
            retVal = self.bounded_affine(self.minPotencial,0,self.maxPotencial,self.maxValue,posPot)
            retVal = retVal - self.bounded_affine(self.minPotencial,0,self.maxPotencial,-self.minValue,negPot)
            return retVal

    def bounded_affine (self,xmin,ymin,xmax,ymax,x):
        k = (ymax-ymin)/(xmax-xmin)
        d = ymin - k * xmin
        y = k * x + d
        if y > ymax:
            y = ymax
        elif y < ymin:
            y = ymin
        return y


    def dumpInterface(self,xmlInterfaces):
        xmlInterface = ET.SubElement(xmlInterfaces, 'interface')
        xmlInterface.set('name',self.name)
        xmlInterface.set('maxPotencial', str(self.maxPotencial))
        xmlInterface.set('minPotencial', str(self.minPotencial))
        xmlInterface.set('maxValue', str(self.maxValue))
        xmlInterface.set('minValue', str(self.minValue))
        xmlInterface.set('value', str(self.value))
        xmlInterface.set('positiveNeuron', self.positiveNeuron.getName())
        xmlInterface.set('negativeNeuron', self.negativeNeuron.getName())
        xmlInterface.set('type', self.type)

    def clone(self):
        toRet=BinaryInterface(self.name,self.positiveNeuron,self.negativeNeuron,self.type)
        toRet.maxPotencial=self.maxPotencial
        toRet.minPotencial=self.minPotencial
        toRet.minValue=self.minValue
        toRet.maxValue=self.maxValue
        toRet.value=self.value
        return toRet


def loadConnection(xmlMI,nn):
    interfacesToReturn={}
    for child in xmlMI:
        miToReturn = BinaryInterface(child.attrib['name'],nn.getNeuron(child.attrib['positiveNeuron']),nn.getNeuron(child.attrib['negativeNeuron']),child.attrib['type'])
        miToReturn.maxPotencial=float(child.attrib['maxPotencial'])
        miToReturn.minPotencial=float(child.attrib['minPotencial'])
        miToReturn.maxValue=float(child.attrib['maxValue'])
        miToReturn.minValue=float(child.attrib['minValue'])
        miToReturn.value=float(child.attrib['value'])
        interfacesToReturn[miToReturn.name]=miToReturn
    return interfacesToReturn

def randonizeModel(model):
     model.addNoise('Weight', 0.5, 26)
     model.addNoise('Vleak', 10, 11)
     model.addNoise('Gleak', 0.2, 11)
     model.addNoise('Sigma', 0.2, 26)
     model.addNoise('Cm', 0.1, 11)

#interfaces for openGym Environment

def envObsToModelObs(envObs):
    observations = []
    for i in range(0, 2):
        observations.append(float(0))
    observations[0] = float(envObs[0])
    observations[1] = float(envObs[1])
    return observations

def modActionToEnvAction(modAction):
    actions = np.zeros(1)
    actions[0] = modAction
    return actions
#interfaces for HyperOpt
def getSpaceForHyperOpt(model):
    return model.neuralnetwork.getSpaceForHyperOpt()


def putVariablesFromHyperOpt(model,vars):
    for neuName in model.neuralnetwork.getNeuronNames():
        deltaGleak=vars[neuName+'_tgl']
            #newValue = self.neuralnetwork.getNeuronTestGleakOfName(neuName) + deltaGleak
            #if newValue < 0.05:
            #    newValue = 0.05
            #elif newValue > 5.0:
            #    newValue = 5.0
            #self.neuralnetwork.setNeuronTestGleakOfName(neuName, newValue)
        model.neuralnetwork.setNeuronTestGleakOfName(neuName, deltaGleak)
        deltaVleak=vars[neuName+'_tvl']
            #newValue = self.neuralnetwork.getNeuronTestVleakOfName(neuName) + deltaVleak
            #if newValue < -90:
            #    newValue = -90
            #elif newValue > 0:
            #    newValue = 0
            #self.neuralnetwork.setNeuronTestVleakOfName(neuName, newValue)
        model.neuralnetwork.setNeuronTestVleakOfName(neuName, deltaVleak)
        deltaCm=vars[neuName+'_tcm']
            #newValue = self.neuralnetwork.getNeuronTestCmOfName(neuName) + deltaCm
            #if newValue < 0.001:
            #    newValue = 0.001
            #elif newValue > 1.0:
            #    newValue = 1.0
            #self.neuralnetwork.setNeuronTestCmOfName(neuName, newValue)
        model.neuralnetwork.setNeuronTestCmOfName(neuName, deltaCm)
    for idx in range(0,len(model.neuralnetwork.connections)):
        deltaWeight=vars['con_'+str(idx)+'_w']
            #newValue = self.neuralnetwork.getConnectionTestWeightOfIdx(idx) + deltaWeight
            #if newValue < 0:
            #    newValue = 0
            #elif newValue > 3.0:
            #    newValue = 3.0
            #self.neuralnetwork.setConnectionTestWeightOfIdx(idx, newValue)
        model.neuralnetwork.setConnectionTestWeightOfIdx(idx, deltaWeight)
        deltaSigma=vars['con_'+str(idx)+'_s']
            #newValue = self.neuralnetwork.getConnectionTestSigmaOfIdx(idx) + deltaSigma
            #if newValue < 0.05:
            #    newValue = 0.05
            #elif newValue > 0.5:
            #    newValue = 0.5
            #self.neuralnetwork.setConnectionTestSigmaOfIdx(idx, newValue)
        model.neuralnetwork.setConnectionTestSigmaOfIdx(idx, deltaSigma)


#Interfaces for PyGad



def getNumGenes():
    return 85

def getGeneType():
    return float

def getGeneSpace():
    return [
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.05, 'high': 5.0}, {'low': -90.0, 'high': 0.0}, {'low': 0.001, 'high': 1.0},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    {'low': 0.0, 'high': 3.0}, {'low': 0.05, 'high': 0.5},
    ]

def getIndividualForPYGAD(model):
    return model.neuralnetwork.getIndividualForPYGAD()


def putIndividualFromPYGAD(model,indiv):
        index = 0
        neuronRandomIndexedNames=model.getNeuronRandomIndexedNames()
        for i in range(0,len(indiv)):
            if i in [0,3,6,9,12,15,18,21,24,27,30]:
                neuronName = neuronRandomIndexedNames[index]
                gleak=indiv[i]
                model.neuralnetwork.setNeuronTestGleakOfName(neuronName,gleak)
            if i in [1,4,7,10,13,16,19,22,25,28,31]:
                neuronName = neuronRandomIndexedNames[index]
                vleak=indiv[i]
                model.neuralnetwork.setNeuronTestVleakOfName(neuronName,vleak)
            if i in [2,5,8,11,14,17,20,23,26,29,32]:
                neuronName = neuronRandomIndexedNames[index]
                cm=indiv[i]
                model.neuralnetwork.setNeuronTestCmOfName(neuronName, cm)
                index=index+1
            if i in [33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83]:
                weight=indiv[i]
                model.neuralnetwork.setConnectionTestWeightOfIdx(index-11, weight)
            if i in [34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84]:
                sigma=indiv[i]
                model.neuralnetwork.setConnectionTestSigmaOfIdx(index-11, sigma)
                index=index+1

