import Models.IFNeuronalCircuit.Connection as conn
import math
import xml.etree.ElementTree as ET
from hyperopt import hp
import Utils.GraphUtils as gu

E_inhib = -90
E_exit = 0
Sigmoid_mu = -40



class Neuron:

    def __init__(self, nname):
        self.neuronName = nname
        self.gleak = 0.0
        self.vleak = 0.0
        self.cm = 0.0
        self.potencial = 0.0
        self.testGleak = 0.0
        self.testVleak = 0.0
        self.testCm = 0.0
        self.testPotencial = 0.0
        self.stateToTrace={'Vleak':[],'Gleak':[],'Cm':[]}
        self.lastActivationTrace = []

    def getStateToTrace(self):
        return self.stateToTrace

    def updateStateToTrace(self):
        self.stateToTrace['Vleak'].append(self.testVleak)
        self.stateToTrace['Gleak'].append(self.testGleak)
        self.stateToTrace['Cm'].append(self.testCm)
        #print('passed by the conection's updateStateToTrace with values for vleak %s'%(self.stateToTrace['Vleak']))

    def __str__(self):
        return "[neu: %s, Pot: %s, TestGleak: %s, TestVleak: %s, TestCm: %s]\n" % (
            self.neuronName, self.potencial, self.testGleak, self.testVleak, self.testCm)
        #self.neuronName, self.potencial, self.gleak, self.vleak, self.cm, self.testGleak, self.testVleak, self.testCm)
        # return "[neu: %s, estado: %s, conexiones: %s]" % (self.neuronName,self.neuronState,len(self.neuronConnections))

    def __repr__(self):
        return "[neu: %s, Gleak: %s, Vleak: %s, Cm: %s, conexiones: %s]" % (
            self.neuronName, self.potencial, self.gleak, self.vleak, self.cm)
        # self.neuronName, self.gleak, self.vleak, self.cm, " ".join(str(c) for c in self.neuronDendrites))

    def __eq__(self, other):
        if not isinstance(other, Neuron):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.neuronName == other.neuronName

    def getName(self):
        return self.neuronName

    # initilizae in quiescent state
    def initialize(self, gleak, vleak, cm):
        self.gleak = gleak
        self.vleak = vleak
        self.cm = cm
        self.potencial = vleak
        self.revertNoise()
        self.testPotencial = self.potencial
        self.revertTest()
        self.lastActivationTrace = []


    def resetLastActivationTrace(self):
        self.lastActivationTrace=[]
    def getLastActivaionTrace(self):
        return self.lastActivationTrace
    def recordActivationTrace(self):
        self.lastActivationTrace.append(self.potencial)


    def setPotencial(self, pot):
        self.potencial = pot
    def getPotencial(self):
        return self.potencial
    def setTestPotencial(self, pot):
        self.testPotencial = pot
    def getTestPotencial(self):
        return self.testPotencial

    def setGleak(self, gl):
        self.gleak = gl
    def getGleak(self):
        return self.gleak
    def setTestGleak(self, gl):
        self.testGleak = gl
    def getTestGleak(self):
        return self.testGleak

    def setVleak(self, vl):
        self.vleak = vl
    def getVleak(self):
        return self.vleak
    def setTestVleak(self, vl):
        self.testVleak = vl
    def getTestVleak(self):
        return self.testVleak

    def setCm(self, cm):
        self.cm = cm
    def getCm(self):
        return self.cm
    def setTestCm(self, cm):
        self.testCm = cm
    def getTestCm(self):
        return self.testCm

    def resetPotencial(self,vleak):
        self.potencial = vleak
        self.testPotencial = vleak

    def computeVnext(self, delta, dendriticConnections):
        numerator = (self.potencial *self.testCm / delta) + (self.testGleak*self.testVleak)
        denominator = self.testCm / delta + self.testGleak
        for dc in dendriticConnections:
            sourcePot = dc.getSource().getPotencial()
            if dc.isType(conn.ConnectionType.ChemEx):
                sig = self.sigmoid(sourcePot, dc.getTestSigma())
                numerator = numerator + dc.getTestWeight()*sig*E_exit
                denominator = denominator + dc.getTestWeight()*sig
            elif dc.isType(conn.ConnectionType.ChemIn):
                sig = self.sigmoid(sourcePot, dc.getTestSigma())
                numerator = numerator + dc.getTestWeight()*sig*E_inhib
                denominator = denominator + dc.getTestWeight()*sig
            else:
                numerator = numerator + dc.getTestWeight()*sourcePot
                denominator = denominator + dc.getTestWeight()
        self.potencial = numerator/denominator


    def sigmoid(self,sourcePot, sigma):
        try:
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        except OverflowError:
            print('source: %s'%(sourcePot))
            print('sigma: %s'%(sigma))
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        return value



    #check if still used
    def isSameAs(self,neu2):
        return self.vleak==neu2.vleak and self.gleak==neu2.gleak and self.cm==neu2.cm and self.testGleak==neu2.testGleak and self.testVleak==neu2.testVleak and self.testCm==neu2.testCm

#needed for Random Seek
    def commitNoise(self):
        self.gleak = self.testGleak
        self.vleak = self.testVleak
        self.cm = self.testCm

    def revertNoise(self):
        self.testGleak = self.gleak
        self.testVleak = self.vleak
        self.testCm = self.cm

 #needed for PYGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testGleak,self.testVleak,self.testCm]
#needed for HeperOpt
    def getVariablesForHyperOpt(self):
        return {
                self.neuronName+'_tgl':hp.uniform(self.neuronName+'_tgl',0.05,5.0),
                self.neuronName + '_tvl': hp.uniform(self.neuronName + '_tvl', -90.0, 0.0),
                self.neuronName + '_tcm': hp.uniform(self.neuronName + '_tcm', 0.001, 1.0)
        }

    def graphVariableTraces(self,folder):
        gu.graphVarTraces(self.stateToTrace,folder,self.neuronName)



def loadNeuron(xmlNeu):
    neuToReturn = Neuron(xmlNeu.attrib['name'])
    neuToReturn.cm=float(xmlNeu.attrib['cm'])
    neuToReturn.gleak=float(xmlNeu.attrib['gleak'])
    neuToReturn.vleak=float(xmlNeu.attrib['vleak'])
    neuToReturn.testCm=float(xmlNeu.attrib['testCm'])
    neuToReturn.testGleak=float(xmlNeu.attrib['testGleak'])
    neuToReturn.testVleak=float(xmlNeu.attrib['testVleak'])
    neuToReturn.potencial=float(xmlNeu.attrib['potencial'])
    neuToReturn.testPotencial=float(xmlNeu.attrib['testPotencial'])
    return neuToReturn




