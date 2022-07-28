import xml.etree.ElementTree as ET
import numpy as np
#import np.random as rng


distortions={}
variances={}

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

    def getPosNeu(self):
        return  self.positiveNeuron
    def getNegNeu(self):
        return  self.negativeNeuron

    def setMaxValue(self,max):
        self.maxValue = max

    def setMinValue(self,min):
        self.minValue = min


    def setValue(self, val):
        self.value = val

    # Was named sync in the original paper (for I&F)
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
            retVal1 = self.bounded_affine(self.minPotencial,0,self.maxPotencial,self.maxValue,posPot)
            retVal2 = self.bounded_affine(self.minPotencial,0,self.maxPotencial,-self.minValue,negPot)
            retVal = retVal1 - retVal2
            #print('retValPos:%s,retValNeg:%s=%s, pot[%s,%s]' % (retVal1, retVal2, retVal1 - retVal2, posPot, negPot))
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



    def isSameAs(self,interface2):
        isSame=True
        isSame=isSame and (self.value==interface2.value)
        isSame=isSame and (self.minValue==interface2.minValue)
        isSame=isSame and (self.maxValue==interface2.maxValue)
        return isSame



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
        model.neuralnetwork.setNeuronTestGleakOfName(neuName, deltaGleak)
        deltaVleak=vars[neuName+'_tvl']
        model.neuralnetwork.setNeuronTestVleakOfName(neuName, deltaVleak)
        deltaCm=vars[neuName+'_tcm']
        model.neuralnetwork.setNeuronTestCmOfName(neuName, deltaCm)
    for idx in range(0,len(model.neuralnetwork.connections)):
        deltaWeight=vars['con_'+str(idx)+'_w']
        model.neuralnetwork.setConnectionTestWeightOfIdx(idx, deltaWeight)
        deltaSigma=vars['con_'+str(idx)+'_s']
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

def randonizeModel(model):
    model.addNoise('Weight', 0.5, 26)
    model.addNoise('Vleak', 10, 11)
    model.addNoise('Gleak', 0.2, 11)
    model.addNoise('Sigma', 0.2, 26)
    model.addNoise('Cm', 0.1, 11)

#Interfaces for Random Seek


def setInitialDistortions():
        global distortions
        distortions['weight']=15
        distortions['vleak'] = 8
        distortions['gleak'] = 8
        distortions['sigma'] = 10
        distortions['cm'] = 10

def setInitialVariance():
    # if needed connections and neurons analize proper values for params a,b,c,d
        global variances
        variances['weight']=0.5
        variances['vleak'] = 8
        variances['gleak'] = 0.2
        variances['sigma'] = 0.2
        variances['cm'] = 0.1

def setBaseDistortions():
    #if needed connections and neurons analize proper values for params a,b,c,d
    global distortions
    distortions['weight']=6
    distortions['vleak'] = 5
    distortions['gleak'] = 4
    distortions['sigma'] = 5
    distortions['cm'] = 4


#def setStepDistortions():
    # global distortions
    # distortions['weight']=np.random.randint(6, distortions['weight']+1)
    # distortions['vleak'] = np.random.randint(4, distortions['vleak']+1)
    # distortions['gleak'] = np.random.randint(4, distortions['gleak']+1)
    # distortions['sigma'] = np.random.randint(5, distortions['sigma']+1)
    # distortions['cm'] = np.random.randint(4, distortions['cm']+1)
    #print('dummy')

def setStepVariance():
        global variances
        variances['weight']=np.random.uniform(0.01, 0.8)
        variances['vleak'] =  np.random.uniform(0.1, 3)
        variances['gleak']  = np.random.uniform(0.05, 0.8)
        variances['sigma'] = np.random.uniform(0.01, 0.08)
        variances['cm'] = np.random.uniform(0.01, 0.3)

def setDecresedDistortions():
    global distortions
    if (distortions['weight'] > 4):
        distortions['weight'] -= 1
    if (distortions['sigma'] > 4):
        distortions['sigma'] -= 1
    if (distortions['vleak'] > 2):
        distortions['vleak'] -= 1
    if (distortions['gleak'] > 2):
        distortions['gleak'] -= 1
    if (distortions['cm'] > 2):
        distortions['cm'] -= 1

def setIncresedDistortions():
    global distortions
    if (distortions['weight'] < 13):
        distortions['weight'] = distortions['weight'] + 1
    if (distortions['sigma'] < 13):
        distortions['sigma'] += 1
    if (distortions['vleak'] < 6):
        distortions['vleak'] += 1
    if (distortions['gleak'] < 6):
        distortions['gleak'] += 1
    if (distortions['cm'] < 6):
        distortions['cm'] += 1



def randonize(model):
    global distortions
    global variances
    model.addNoise('Weight', variances['weight'], distortions['weight'])
    model.addNoise('Vleak', variances['vleak'], distortions['vleak'])
    model.addNoise('Gleak', variances['gleak'], distortions['gleak'])
    model.addNoise('Sigma', variances['sigma'], distortions['sigma'])
    model.addNoise('Cm', variances['cm'], distortions['cm'])

def randozieAll(model):
        model.addNoise('Weight', 0.5, 26)
        model.addNoise('Vleak', 10, 11)
        model.addNoise('Gleak', 0.2, 11)
        model.addNoise('Sigma', 0.2, 26)
        model.addNoise('Cm', 0.1, 11)

