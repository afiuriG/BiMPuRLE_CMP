import enum
import Utils.GraphUtils as gu
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
        self.stateToTrace={'Weight':[],'Sigma':[]}

    def updateStateToTrace(self):
        self.stateToTrace['Weight'].append(self.testWeight)
        self.stateToTrace['Sigma'].append(self.testSigma)
        #print('passed by the conection's updateStateToTrace with values for weight %s' %(self.stateToTrace['Weight']))


    def __str__(self):
        return "<Con from:%s, To: %s, W: %s, S: %s, TestW: %s, TestS: %s>\n" % (self.connSource.getName(), self.connTarget.getName(), self.connWeight, self.connSigma, self.testWeight, self.testSigma)

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
    def setSource(self,src):
        self.connSource=src

    def getTarget(self):
        return self.connTarget
    def setTarget(self,tar):
        self.connTarget=tar

    def graphVariableTraces(self,folder):
        gu.graphVarTraces(self.stateToTrace,folder,self.connSource.getName()+'/'+self.connTarget.getName())


    def isSameAs(self,con2):
        return self.testWeight==con2.testWeight and self.testSigma==con2.testSigma

#needed for PyGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testWeight,self.testSigma]

#needed for HyperOpt
    def getSpaceForHyperOpt(self,index):
        namew='con_'+str(index)+'_w'
        names='con_'+str(index)+'_s'
        return {
                namew: hp.uniform(namew, 0.0, 3.0),
                names: hp.uniform(names, 0.05, 0.5),
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

