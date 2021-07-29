import Models.IFNeuronalCircuit.Connection as conn

import math
import xml.etree.ElementTree as ET
import random
import numpy as np
from hyperopt import hp

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

    # se inicializa en estado de reposo
    def initialize(self, gleak, vleak, cm):
        self.gleak = gleak
        self.vleak = vleak
        self.cm = cm
        self.potencial = vleak
        self.revertTest()


    def commitTest(self):
        self.gleak = self.testGleak
        self.vleak = self.testVleak
        self.cm = self.testCm
        self.potencial = self.testPotencial

    def revertTest(self):
        self.testGleak = self.gleak
        self.testVleak = self.vleak
        self.testCm = self.cm
        self.testPotencial = self.potencial

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
        #self.testPotencial = numerator/denominator
        self.potencial = numerator/denominator


    def sigmoid(self,sourcePot, sigma):
        try:
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        except OverflowError:
            print('source: %s'%(sourcePot))
            print('sigma: %s'%(sigma))
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        #return 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        return value

    def commitNoise(self):
        self.gleak = self.testGleak
        self.vleak = self.testVleak
        self.cm = self.testCm

    def revertNoise(self):
        self.testGleak = self.gleak
        self.testVleak = self.vleak
        self.testCm = self.cm


    def dumpNeuron(self,xmlNeurons):
        xmlNeuron = ET.SubElement(xmlNeurons, 'neuron')
        xmlNeuron.set('name',self.neuronName)
        xmlNeuron.set('gleak',str(self.gleak))
        xmlNeuron.set('vleak',str(self.vleak))
        xmlNeuron.set('cm',str(self.cm))
        xmlNeuron.set('potencial',str(self.potencial))
        xmlNeuron.set('testGleak',str(self.testGleak))
        xmlNeuron.set('testVleak',str(self.testVleak))
        xmlNeuron.set('testCm',str(self.testCm))
        xmlNeuron.set('testPotencial',str(self.testPotencial))

    def getComponentOfIndividualForPYGAD(self):
        return [self.testGleak,self.testVleak,self.testCm]

    def getVariablesForHyperOpt(self):
        return {
                self.neuronName+'_tgl':hp.uniform(self.neuronName+'_tgl',0.05,5.0),
                self.neuronName + '_tvl': hp.uniform(self.neuronName + '_tvl', -90.0, 0.0),
                self.neuronName + '_tcm': hp.uniform(self.neuronName + '_tcm', 0.001, 1.0)
            #0.2,10.0,0.1
                #self.neuronName + '_tgl': hp.normal(self.neuronName + '_tgl', 0.0, 1.0),
                #self.neuronName + '_tvl': hp.normal(self.neuronName + '_tvl', 0.0, 50.0),
                #self.neuronName + '_tcm': hp.normal(self.neuronName + '_tcm', 0.0, 0.5)

        }

    def isSameAs(self,neu2):
        return self.testGleak==neu2.testGleak and self.testVleak==neu2.testVleak and self.testCm==neu2.testCm

    def clone(self):
        neuToRet=Neuron(self.neuronName)
        neuToRet.gleak=self.gleak
        neuToRet.vleak=self.vleak
        neuToRet.cm=self.cm
        neuToRet.potencial=self.potencial
        neuToRet.testGleak=self.testGleak
        neuToRet.testVleak=self.testVleak
        neuToRet.testCm=self.testCm
        neuToRet.testPotencial=self.testPotencial
        return neuToRet





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

def combine(neu1,neu2,child1,child2):
    newGleak = 0
    newVleak = 0
    newCm = 0
    combinacionCase=random.randint(1,6)
    if combinacionCase==1:
        newGleak1 = neu1.testGleak
        newVleak1 = neu2.testVleak
        newCm1 = neu2.testCm
        newGleak2 = neu2.testGleak
        newVleak2 = neu1.testVleak
        newCm2 = neu1.testCm
    elif combinacionCase==2:
        newGleak1 = neu2.testGleak
        newVleak1 = neu1.testVleak
        newCm1 = neu2.testCm
        newGleak2 = neu1.testGleak
        newVleak2 = neu2.testVleak
        newCm2 = neu1.testCm
    elif combinacionCase == 3:
        newGleak1 = neu2.testGleak
        newVleak1 = neu2.testVleak
        newCm1 = neu1.testCm
        newGleak2 = neu1.testGleak
        newVleak2 = neu1.testVleak
        newCm2 = neu2.testCm
    elif combinacionCase==4:
        newGleak1 = neu2.testGleak
        newVleak1 = neu1.testVleak
        newCm1 = neu1.testCm
        newGleak2 = neu1.testGleak
        newVleak2 = neu2.testVleak
        newCm2 = neu2.testCm
    elif combinacionCase == 5:
        newGleak1 = neu1.testGleak
        newVleak1 = neu2.testVleak
        newCm1 = neu1.testCm
        newGleak2 = neu2.testGleak
        newVleak2 = neu1.testVleak
        newCm2 = neu2.testCm
    elif combinacionCase==6:
        newGleak1 = neu1.testGleak
        newVleak1 = neu1.testVleak
        newCm1 = neu2.testCm
        newGleak2 = neu2.testGleak
        newVleak2 = neu2.testVleak
        newCm2 = neu1.testCm
    child1.testGleak = newGleak1
    child1.testVleak = newVleak1
    child1.testCm = newCm1
    child2.testGleak = newGleak2
    child2.testVleak = newVleak2
    child2.testCm = newCm2




# not used
def mutate(neu):
    what = random.randint(1, 3)
    if what ==1:
        #mutate GLeak
        normalDistribuitedValue = np.random.normal(0, 0.2)
        newVal=neu.testGleak+normalDistribuitedValue
        if newVal < 0.05:
            newVal = 0.05
        elif newVal > 5.0:
            newVal = 5.0
        neu.testGleak=newVal
        #print('muto Gleak: %s' % (normalDistribuitedValue))
    elif what==2:
        # mutate VLeak
        normalDistribuitedValue = np.random.normal(0, 10)
        newVal=neu.testVleak+normalDistribuitedValue
        if newVal < -90:
            newVal = -90
        elif newVal > 0:
            newVal = 0
        neu.testVleak = newVal
        #print('muto Vleak: %s' % (normalDistribuitedValue))
    elif what == 3:
        # mutate Cm
        normalDistribuitedValue = np.random.normal(0, 0.1)
        newVal=neu.testCm+normalDistribuitedValue
        if newVal < 0.001:
            newVal = 0.001
        elif newVal > 1.0:
            newVal = 1.0
        neu.testCm = newVal
        #print('muto Cm: %s' % (normalDistribuitedValue))


