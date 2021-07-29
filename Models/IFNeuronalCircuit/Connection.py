import enum
import xml.etree.ElementTree as ET
import random
import numpy as np
from hyperopt import hp

class Connection:

    def __init__(self, contype, source, target, weight, sigma):
        self.connType = contype
        self.connSource = source
        self.connTarget = target
        self.connWeight = weight
        self.connSigma = sigma
        self.testWeight = weight
        self.testSigma = sigma

    def __str__(self):
        return "<Con from:%s, To: %s, W: %s, S: %s, TestW: %s, TestS: %s>" % (self.connSource.getName(), self.connTarget.getName(), self.connWeight, self.connSigma, self.testWeight, self.testSigma)

    def __repr__(self):
        return "<Con from:%s, To: %s, W: %s, S: %s>" % (self.connSource.getName(), self.connTarget.getName(), self.connWeight, self.connSigma)

    def setWeight(self, w):
        self.connWeight = w

    def getWeight(self):
        return self.connWeight

    def setSigma(self, sigma):
        self.connSigma = sigma

    def getSigma(self):
        return self.connSigma

    def setTestWeight(self, w):
        self.testWeight = w

    def getTestWeight(self):
        return self.testWeight

    def setTestSigma(self, sigma):
        self.testSigma = sigma

    def getTestSigma(self):
        return self.testSigma

    def isType(self, type):
        return type == self.connType



    def getSource(self):
        return self.connSource

    def getTarget(self):
        return self.connTarget


    def commitNoise(self):
        self.connWeight = self.testWeight
        self.connSigma = self.testSigma


    def revertNoise(self):
        self.testWeight = self.connWeight
        self.testSigma = self.connSigma

    def dumpConnection(self,xmlConnections):
        xmlConnection = ET.SubElement(xmlConnections, 'connection')
        xmlConnection.set('connType',str(self.connType))
        xmlConnection.set('connSource',self.connSource.getName())
        xmlConnection.set('connTarget',self.connTarget.getName())
        xmlConnection.set('connWeight',str(self.connWeight))
        xmlConnection.set('connSigma',str(self.connSigma))
        xmlConnection.set('testWeight',str(self.testWeight))
        xmlConnection.set('testSigma',str(self.testSigma))

    def getComponentOfIndividualForPYGAD(self):
        return [self.testWeight,self.testSigma]

    def getSpaceForHyperOpt(self,index):
        namew='con_'+str(index)+'_w'
        names='con_'+str(index)+'_s'
        return {
                namew: hp.uniform(namew, 0.0, 3.0),
                names: hp.uniform(names, 0.05, 0.5),
                #0,5,0.2
                #namew: hp.normal(namew, 0.0, 2.5),
                #names: hp.normal(names, 0.0, 1.0)
        }

    def isSameAs(self,con2):
        return self.testWeight==con2.testWeight and self.testSigma==con2.testSigma

    def clone(self):
        conToRet = Connection(self.connType,self.connSource,self.connTarget,self.connWeight,self.connSigma)
        conToRet.testWeight=self.testWeight
        conToRet.testSigma=self.testSigma
        return conToRet

def loadConnection(xmlCon,nn):
    connToReturn = Connection(connTypeFromStr(xmlCon.attrib['connType']),nn.getNeuron(xmlCon.attrib['connSource']),nn.getNeuron(xmlCon.attrib['connTarget']),float(xmlCon.attrib['connWeight']),float(xmlCon.attrib['connSigma']))
    connToReturn.testWeight=float(xmlCon.attrib['testWeight'])
    connToReturn.testSigma=float(xmlCon.attrib['testSigma'])
    return connToReturn

def combine(con1,con2,host1,host2):
    if random.random() <= 0.5:
        newWeight1 = con1.testWeight
        newSigma1 = con2.testSigma
        newWeight2 = con2.testWeight
        newSigma2 = con1.testSigma
    else:
        newWeight1 = con2.testWeight
        newSigma1 = con1.testSigma
        newWeight2 = con1.testWeight
        newSigma2 = con2.testSigma
    host1.setTestWeight=newWeight1
    host1.setTestSigma=newSigma1
    host2.setTestWeight=newWeight2
    host2.setTestSigma=newSigma2

def mutate(conn):
        what = random.randint(1, 2)
        if what == 1:
            # mutate Weight
            normalDistribuitedValue = np.random.normal(0, 0.5)
            newVal = conn.testWeight + normalDistribuitedValue
            if newVal < 0:
                newVal = 0
            elif newVal > 3.0:
                newVal = 3.0
            conn.testWeight = newVal
            #print(conn)
            #print('muto weight: %s' % (normalDistribuitedValue))
        else:
            # mutate Sigma
            normalDistribuitedValue = np.random.normal(0,0.2)
            newVal = conn.testSigma + normalDistribuitedValue
            if newVal < 0.05:
                newVal = 0.05
            elif newVal > 0.5:
                newVal = 0.5
            conn.testSigma = newVal
            #print(conn)
            #print('muto sigma: %s' % (normalDistribuitedValue))


class ConnectionType(enum.Enum):
    ChemIn = 1
    ChemEx = 2
    SGJ = 3
    AGJ = 4

def connTypeFromStr(str):
    if str=='ConnectionType.ChemIn':
        return ConnectionType.ChemIn
    if str=='ConnectionType.ChemEx':
        return ConnectionType.ChemEx
    if str=='ConnectionType.AGJ':
        return ConnectionType.AGJ
    if str=='ConnectionType.SGJ':
        return ConnectionType.SGJ

