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
        return "<%s,%s,%s,%s,%s,%s>\n" % (self.connSource.getName(), self.connTarget.getName(), self.testWeight, self.testSigma,self.connWeight, self.connSigma)

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



    def isSameAs(self,con2):
        return self.testWeight==con2.testWeight and self.testSigma==con2.testSigma and self.connWeight==con2.connWeight and self.connSigma==con2.connSigma


#needed for PyGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testWeight,self.testSigma]

#needed for HyperOpt
    def getSpaceForHyperOpt(self,index):
        namew='con_'+str(index)+'_w'
        names='con_'+str(index)+'_s'
        return {
                namew: hp.uniform(namew, 0.0, 5.0),
                names: hp.uniform(names, 0.0, 1.0),
                #namew: hp.normal(namew, 0.0, 2.5),
                #names: hp.normal(names, 0.0, 1.0)
        }

#needed for Random Seek
    def commitNoise(self):
        self.connWeight = self.testWeight
        self.connSigma = self.testSigma


    def revertNoise(self):
        self.testWeight = self.connWeight
        self.testSigma = self.connSigma





def loadConnection(xmlCon,nn):
    connToReturn = Connection(connTypeFromStr(xmlCon.attrib['connType']),nn.getNeuron(xmlCon.attrib['connSource']),nn.getNeuron(xmlCon.attrib['connTarget']),float(xmlCon.attrib['connWeight']),float(xmlCon.attrib['connSigma']))
    connToReturn.testWeight=float(xmlCon.attrib['testWeight'])
    connToReturn.testSigma=float(xmlCon.attrib['testSigma'])
    return connToReturn




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

