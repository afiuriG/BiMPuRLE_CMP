import xml.etree.ElementTree as ET
import numpy as np
import Models.Haimovici.Neuron as neu


distortions={}
variances={}

class BinaryInterface:

    def __init__(self, name, posNeu, negNeu, type):
        self.name = name
        self.valleyVal = 0
        self.value=0
        self.positiveNeuron = posNeu
        self.negativeNeuron = negNeu
        self.type = type

    def __str__(self):
        return "<%s,%s,%s,%s,%s>\n" % (
            self.name, self.type, self.positiveNeuron.getName(), self.negativeNeuron.getName(), self.valleyVal)

    def __repr__(self):
        return "[pos: %s, neg: %s, value: %s]" % (
            self.positiveNeuron, self.negativeNeuron, self.state)

    def reset(self):
        self.value = 0

    def getName(self):
        return  self.name


    def setValue(self, val):
        self.value = val


    #Was named sync in the original paper (for I&F)
    def feedNN(self):
        if self.type == 'IN':
            if self.value>=self.valleyVal:
                self.positiveNeuron.setState(neu.NeuronState.Excited)
                self.negativeNeuron.setState(neu.NeuronState.Quiescent)
                #print (self.value,posPot,self.minPotencial)
            else:
                self.positiveNeuron.setState(neu.NeuronState.Quiescent)
                self.negativeNeuron.setState(neu.NeuronState.Excited)
                #print(self.value, self.minPotencial,negPot)

    def getFeedBackNN(self):
        randomNoise = np.random.uniform(0.0, 0.01, 2)
        posacel=0.2#+randomNoise[0]
        negacel=0.1#+randomNoise[1]
        actionToReturn=0
        if self.type == 'OUT':
            negState = self.negativeNeuron.getState()
            posState = self.positiveNeuron.getState()
            if negState==neu.NeuronState.Excited:
                actionToReturn=actionToReturn-negacel
            if posState==neu.NeuronState.Excited:
                actionToReturn=actionToReturn+posacel
            #print ('retValPos:%s,retValNeg:%s=%s, pot[%s,%s]'%(retVal1,retVal2,retVal1-retVal2,posPot,negPot))
            return actionToReturn


    def isSameAs(self,interface2):
        isSame=True
        isSame=isSame and (self.value==interface2.value)
        isSame=isSame and (self.valleyVal==interface2.valleyVal)
        return isSame

def loadConnection(xmlMI,nn):
    interfacesToReturn={}
    for child in xmlMI:
        miToReturn = BinaryInterface(child.attrib['name'],nn.getNeuron(child.attrib['positiveNeuron']),nn.getNeuron(child.attrib['negativeNeuron']),child.attrib['type'])
        miToReturn.valleyVal=float(child.attrib['valleyVal'])
        miToReturn.value=float(child.attrib['value'])
        interfacesToReturn[miToReturn.name]=miToReturn
    return interfacesToReturn

####################################################
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
    #uncomment when to learn neuron parameters is needed
    for neuName in model.neuralnetwork.getNeuronNames():
        val_tth=vars[neuName+'_tth']
        model.neuralnetwork.setNeuronTestThresholdOfName(neuName, val_tth)
        val_tr1=vars[neuName+'_tr1']
        model.neuralnetwork.setNeuronTestR1OfName(neuName, val_tr1)
        val_tr2=vars[neuName+'_tr2']
        model.neuralnetwork.setNeuronTestR2OfName(neuName, val_tr2)
    for idx in range(0,len(model.neuralnetwork.connections)):
        deltaWeight=vars['con_'+str(idx)+'_w']
        model.neuralnetwork.setConnectionTestWeightOfIdx(idx, deltaWeight)


#Interfaces for PyGad


def getNumGenes():
    #only connections
    #return 52
    return 96
    #connections and neurons

def getGeneType():
    return float

def getGeneSpace():
    #for connections and neurons uncomment the commented lines
    low_th=0.0
    high_th=10.0
    low_r1=0.0
    high_r1=0.1
    low_r2=0.0
    high_r2=1.0
    low_w=0.0
    high_w=10.0


    return [
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.5, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.5, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.5, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 0.001}, {'low': 0.50, 'high': 0.9},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
        {'low': 0.0, 'high': 10.0},
            ]


def getIndividualForPYGAD(model):
    return model.neuralnetwork.getIndividualForPYGAD()


def putIndividualFromPYGAD(model,indiv):
    #for connections only
    #putIndividualFromPYGAD_OnlyConnMode(model, indiv)
    #for connections and neurons
    putIndividualFromPYGAD_ConnNeuMode(model,indiv)


def putIndividualFromPYGAD_OnlyConnMode(model,indiv):
    index = 0
    neuronRandomIndexedNames = model.getNeuronRandomIndexedNames()
    for i in range(0, len(indiv)):
        if i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
                 50]:
            weight = indiv[i]
            model.neuralnetwork.setConnectionTestWeightOfIdx(index, weight)
        if i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
                 51]:
            sigma = indiv[i]
            model.neuralnetwork.setConnectionTestSigmaOfIdx(index, sigma)
            index = index + 1

def putIndividualFromPYGAD_ConnNeuMode(model, indiv):
    index = 0
    neuronRandomIndexedNames = model.getNeuronRandomIndexedNames()
    for i in range(0, len(indiv)):
        if i in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]:
            neuronName = neuronRandomIndexedNames[index]
            paramA = indiv[i]
            model.neuralnetwork.setNeuronTestThresholdOfName(neuronName, paramA)
        if i in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]:
            neuronName = neuronRandomIndexedNames[index]
            paramB = indiv[i]
            model.neuralnetwork.setNeuronTestR1OfName(neuronName, paramB)
        if i in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]:
            neuronName = neuronRandomIndexedNames[index]
            paramD = indiv[i]
            model.neuralnetwork.setNeuronTestR2OfName(neuronName, paramD)
            index = index + 1
        if i in [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]:
            weight = indiv[i]
            model.neuralnetwork.setConnectionTestWeightOfIdx(index - 11, weight)
            index = index + 1


def randonizeModel(model):
    model.addNoise('Weight', 1.0, 26)
    model.addNoise('Th', 0.49, 11)
    model.addNoise('R1', 0.025, 11)
    model.addNoise('R2', 10.0, 11)

#Interfaces for Random Seek


def setInitialDistortions():
        global distortions
        distortions['Weight']=15
        distortions['Th']=8
        distortions['R1']=8
        distortions['R2']=8

def setInitialVariance():
        global variances
        variances['Weight']=1.0
        variances['Th'] = 0.49
        variances['R1'] = 0.025
        variances['R2'] = 10



def setBaseDistortions():
    global distortions
    distortions['Weight']=6
    distortions['Th'] = 5
    distortions['R1'] = 5
    distortions['R2'] = 5


    #def setStepDistortions():
    #global distortions
    #distortions['weight']=np.random.randint(0, distortions['weight']+1)
    #distortions['a'] = np.random.randint(0, distortions['a']+1)
    #distortions['b'] = np.random.randint(0, distortions['b']+1)
    #distortions['c'] = np.random.randint(0, distortions['c']+1)
    #distortions['d'] = np.random.randint(0, distortions['d']+1)
    #distortions['sigma'] = np.random.randint(0, distortions['sigma']+1)
    #print('dummy step distortion')

def setStepVariance():
        global variances
        variances['Weight']=np.random.uniform(0.01, 0.8)
        variances['Th']=np.random.uniform(0.05, 0.49)
        variances['R1']=np.random.uniform(0.002, 0.025)
        variances['R2']=np.random.uniform(1, 10)

def setDecresedDistortions():
    global distortions
    if (distortions['Weight'] > 4):
        distortions['Weight'] -= 1
    if (distortions['Th'] > 2):
        distortions['Th'] -= 1
    if (distortions['R1'] > 2):
        distortions['R1'] -= 1
    if (distortions['R2'] > 2):
        distortions['R2'] -= 1




def setIncresedDistortions():
    global distortions
    if (distortions['Weight'] < 13):
        distortions['Weight'] += 1
    if (distortions['Th'] < 6):
        distortions['Th'] += 1
    if (distortions['R1'] < 6):
        distortions['R1'] += 1
    if (distortions['R2'] < 6):
        distortions['R2'] += 1




def randonize(model):
    global distortions
    global variances
    model.addNoise('Weight', variances['Weight'], distortions['Weight'])
    model.addNoise('Th', variances['Th'], distortions['Th'])
    model.addNoise('R1', variances['R1'], distortions['R1'])
    model.addNoise('R2', variances['R2'], distortions['R2'])

def randozieAll(model):
    model.addNoise('Weight', 1, 26)
    model.addNoise('Th', 0.49, 11)
    model.addNoise('R1', 0.025, 11)
    model.addNoise('R2', 10, 11)

