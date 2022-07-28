import xml.etree.ElementTree as ET
import numpy as np
#import np.random as rng


distortions={}
variances={}

class BinaryInterface:

    def __init__(self, name, posNeu, negNeu, type):
        self.name = name
        self.maxPotencial =20
        self.minPotencial = -20
        self.valleyVal = 0
        self.minValue = 0
        self.maxValue=0
        self.value = 0
        self.positiveNeuron = posNeu
        self.negativeNeuron = negNeu
        self.type = type

    def __str__(self):
        return "<%s,%s,%s,%s,%s>\n" % (
            self.name, self.type, self.positiveNeuron.getName(), self.negativeNeuron.getName(), self.value)

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

    def resetFired(self):
        self.negativeNeuron.resetFired()
        self.positiveNeuron.resetFired()

    #Was named sync in the original paper (for I&F)
    def feedNN(self):
        if self.type == 'IN':
            if self.value>=self.valleyVal:
                corVal = self.value/self.maxValue
                posPot = (self.maxPotencial-self.minPotencial)*corVal+self.minPotencial
                self.positiveNeuron.setPotencial(posPot)
                self.negativeNeuron.setPotencial(self.minPotencial)
                self.positiveNeuron.resetRecovery()
                self.negativeNeuron.resetRecovery()
                #print (self.value,posPot,self.minPotencial)
            else:
                corVal = self.value/(-self.minValue)
                negPot = (self.maxPotencial-self.minPotencial)*(-corVal)+self.minPotencial
                self.positiveNeuron.setPotencial(self.minPotencial)
                self.negativeNeuron.setPotencial(negPot)
                self.positiveNeuron.resetRecovery()
                self.negativeNeuron.resetRecovery()
                #print(self.value, self.minPotencial,negPot)

    def getFeedBackNN(self):
        if self.type == 'OUT':
            negPot = self.negativeNeuron.getPotencial()
            posPot = self.positiveNeuron.getPotencial()
            retVal1 = self.bounded_affine(self.minPotencial, 0, self.maxPotencial, self.maxValue, posPot)
            retVal2 = self.bounded_affine(self.minPotencial, 0, self.maxPotencial, -self.minValue, negPot)
            #print ('retValPos:%s,retValNeg:%s=%s, pot[%s,%s]'%(retVal1,retVal2,retVal1-retVal2,posPot,negPot))
            return retVal1-retVal2

    def bounded_affine(self, xmin, ymin, xmax, ymax, x):
        a = (ymax - ymin) / (xmax - xmin)
        d = ymin - a * xmin
        y = a * x + d
        if y > ymax:
            y = ymax
        elif y < ymin:
            y = ymin
        return y



    # def clone(self):
    #     toRet=BinaryInterface(self.name,self.positiveNeuron,self.negativeNeuron,self.type)
    #     toRet.valleyVal=self.valleyVal
    #     toRet.value=self.value
    #     toRet.minValue=self.minValue
    #     toRet.maxValue=self.maxValue
    #     return toRet

    def isSameAs(self,interface2):
        isSame=True
        isSame=isSame and (self.value==interface2.value)
        isSame=isSame and (self.minValue==interface2.minValue)
        isSame=isSame and (self.maxValue==interface2.maxValue)
        isSame=isSame and (self.valleyVal==interface2.valleyVal)
        return isSame

def loadConnection(xmlMI,nn):
    interfacesToReturn={}
    for child in xmlMI:
        miToReturn = BinaryInterface(child.attrib['name'],nn.getNeuron(child.attrib['positiveNeuron']),nn.getNeuron(child.attrib['negativeNeuron']),child.attrib['type'])
        miToReturn.valleyVal=float(child.attrib['valleyVal'])
        miToReturn.value=float(child.attrib['value'])
        miToReturn.minValue=float(child.attrib['minVal'])
        miToReturn.maxValue=float(child.attrib['maxVal'])
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
        deltaA=vars[neuName+'_ta']
        model.neuralnetwork.setNeuronTestParamAOfName(neuName, deltaA)
        deltaB=vars[neuName+'_tb']
        model.neuralnetwork.setNeuronTestParamBOfName(neuName, deltaB)
        deltaC=vars[neuName+'_tc']
        model.neuralnetwork.setNeuronTestParamCOfName(neuName, deltaC)
        deltaD=vars[neuName+'_td']
        model.neuralnetwork.setNeuronTestParamDOfName(neuName, deltaD)
    for idx in range(0,len(model.neuralnetwork.connections)):
        deltaWeight=vars['con_'+str(idx)+'_w']
        model.neuralnetwork.setConnectionTestWeightOfIdx(idx, deltaWeight)
        deltaSigma=vars['con_'+str(idx)+'_s']
        model.neuralnetwork.setConnectionTestSigmaOfIdx(idx, deltaSigma)


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
    return [
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.019999, 'high': 0.020001}, {'low': 0.19999, 'high': 0.20001}, {'low': -21.0, 'high': -20.0},{'low': 7.0, 'high': 8.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
        {'low': 0.0, 'high': 5.0}, {'low': 0.05, 'high': 1.0},
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
        if i in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]:
            neuronName = neuronRandomIndexedNames[index]
            paramA = indiv[i]
            model.neuralnetwork.setNeuronTestParamAOfName(neuronName, paramA)
        if i in [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]:
            neuronName = neuronRandomIndexedNames[index]
            paramB = indiv[i]
            model.neuralnetwork.setNeuronTestParamBOfName(neuronName, paramB)
        if i in [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]:
            neuronName = neuronRandomIndexedNames[index]
            paramC = indiv[i]
            model.neuralnetwork.setNeuronTestParamCOfName(neuronName, paramC)
        if i in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]:
            neuronName = neuronRandomIndexedNames[index]
            paramD = indiv[i]
            model.neuralnetwork.setNeuronTestParamDOfName(neuronName, paramD)
            index = index + 1
        if i in [44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92,
                 94]:
            weight = indiv[i]
            model.neuralnetwork.setConnectionTestWeightOfIdx(index - 11, weight)
        if i in [45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93,
                 95]:
            sigma = indiv[i]
            model.neuralnetwork.setConnectionTestSigmaOfIdx(index - 11, sigma)
            index = index + 1


def randonizeModel(model):
    model.addNoise('Weight', 1.0, 26)
    model.addNoise('Sigma', 0.5, 26)
    model.addNoise('a', 0.49, 11)
    model.addNoise('b', 0.025, 11)
    model.addNoise('c', 10.0, 11)
    model.addNoise('d', 3.0, 11)

#Interfaces for Random Seek


def setInitialDistortions():
        global distortions
        distortions['weight']=15
        distortions['a']=8
        distortions['b']=8
        distortions['c']=8
        distortions['d']=8
        distortions['sigma'] = 10

def setInitialVariance():
        global variances
        variances['weight']=1.0
        variances['sigma'] = 0.25
        variances['a'] = 0.49
        variances['b'] = 0.025
        variances['c'] = 10
        variances['d'] = 3



def setBaseDistortions():
    global distortions
    distortions['weight']=6
    distortions['sigma'] = 5
    distortions['a'] = 6
    distortions['b'] = 6
    distortions['c'] = 6
    distortions['d'] = 6


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
        variances['weight']=np.random.uniform(0.01, 0.8)
        variances['sigma'] = np.random.uniform(0.01, 0.08)
        variances['a']=np.random.uniform(0.05, 0.49)
        variances['b']=np.random.uniform(0.002, 0.025)
        variances['c']=np.random.uniform(1, 10)
        variances['d']=np.random.uniform(0.3, 3)

def setDecresedDistortions():
    global distortions
    if (distortions['weight'] > 4):
        distortions['weight'] -= 1
    if (distortions['sigma'] > 4):
        distortions['sigma'] -= 1
    if (distortions['a'] > 2):
        distortions['a'] -= 1
    if (distortions['b'] > 2):
        distortions['b'] -= 1
    if (distortions['c'] > 2):
        distortions['c'] -= 1
    if (distortions['d'] > 2):
        distortions['d'] -= 1




def setIncresedDistortions():
    global distortions
    if (distortions['weight'] < 13):
        distortions['weight'] += 1
    if (distortions['sigma'] < 13):
        distortions['sigma'] += 1
    if (distortions['a'] < 6):
        distortions['a'] += 1
    if (distortions['b'] < 6):
        distortions['b'] += 1
    if (distortions['c'] < 6):
        distortions['c'] += 1
    if (distortions['d'] < 6):
        distortions['d'] += 1




def randonize(model):
    global distortions
    global variances
    model.addNoise('Weight', variances['weight'], distortions['weight'])
    model.addNoise('a', variances['a'], distortions['a'])
    model.addNoise('b', variances['b'], distortions['b'])
    model.addNoise('c', variances['c'], distortions['c'])
    model.addNoise('d', variances['d'], distortions['d'])
    model.addNoise('Sigma', variances['sigma'], distortions['sigma'])

def randozieAll(model):
    model.addNoise('Weight', 1, 26)
    model.addNoise('Sigma', 0.5, 26)
    model.addNoise('a', 0.49, 11)
    model.addNoise('b', 0.025, 11)
    model.addNoise('c', 10, 11)
    model.addNoise('d', 3, 11)

