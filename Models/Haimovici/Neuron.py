import enum
import numpy as np
from hyperopt import hp
import Utils.GraphUtils as gu
import Models.Haimovici.Connection as con
import Utils.GraphUtils as gu


class Neuron:

#Pensar un poco como es la inicializacion
    def __init__(self, nname):
        self.neuronName = nname
        self.state= None
        self.bufferState=None
        self.threshold=0
        self.testThreshold=0
        #prob of self exitation
        self.r1=0.001
        #prob of get out of R
        self.r2=0.1
        self.testR1=0.001
        self.testR2=0.1

        self.lastActivationTrace = []
        self.stateToTrace = {'Th': [], 'R1': [], 'R2': []}


    def __str__(self):
        return "[neu: %s, State: %s, Threshold: %s, R1: %s, R2: %s]\n" % (
            self.neuronName, self.state,self.threshold,self.r1,self.r2)

    def __repr__(self):
        return "[neu: %s, State: %s, Threshold: %s]" % (
            self.neuronName, self.state, self.threshold)

    def __eq__(self, other):
        if not isinstance(other, Neuron):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.neuronName == other.neuronName

    def getName(self):
        return self.neuronName

    # initialized in Quiescent stare
    def initialize(self, th, r1,r2):
        self.state=NeuronState.Quiescent
        self.bufferState=NeuronState.Quiescent
        self.threshold = th
        self.testThreshold = th
        self.testR1=r1
        self.r1=r1
        self.testR2=r2
        self.r2=r2
        self.lastActivationTrace = []

    def getStateToTrace(self):
        return self.stateToTrace

    def updateStateToTrace(self):
        self.stateToTrace['Th'].append(self.testThreshold)
        self.stateToTrace['R1'].append(self.testR1)
        self.stateToTrace['R2'].append(self.testR2)
        #print('passed by the conection's updateStateToTrace with values for vleak %s'%(self.stateToTrace['Vleak']))

    def graphVariableTraces(self,folder):
        gu.graphVarTraces(self.stateToTrace,folder,self.neuronName)

    def resetLastActivationTrace(self):
        self.lastActivationTrace=[]
    def getLastActivaionTrace(self):
        return self.lastActivationTrace
    def recordActivationTrace(self):
        self.lastActivationTrace.append(str(self.state))

    def setState(self, st):
        self.state = st
    def getState(self):
        return self.state
    def setBufferState(self, st):
        self.bufferState = st
    def getBufferState(self):
        return self.bufferState

    def commitComputation(self):
        self.state=self.bufferState

    def setThreshold(self, th):
        self.threshold = th
    def getThreshold(self):
        return self.threshold
    def setTestThreshold(self, th):
        self.testThreshold = th
    def getTestThreshold(self):
        return self.testThreshold


    def setR1(self, r1):
        self.r1 = r1
    def getR1(self):
        return self.r1
    def setTestR1(self, r1):
        self.testR1 = r1
    def getTestR1(self):
        return self.testR1

    def setR2(self, r2):
        self.r2 = r2
    def getR2(self):
        return self.r2
    def setTestR2(self, r2):
        self.testR2 = r2
    def getTestR2(self):
        return self.testR2

    # GreemberAndHasting compute vnext
    def computeVnextGandH(self, dendriticConnections):
        if not len(dendriticConnections) == 0:
            if self.state == NeuronState.Refractary:
               self.bufferState = NeuronState.Quiescent
            elif self.state == NeuronState.Excited:
                self.bufferState = NeuronState.Refractary
            elif self.state == NeuronState.Quiescent:
                atLeatsOne=False
                for co in dendriticConnections:
                    sourceState = co.getSource().getState()
                    atLeatsOne=atLeatsOne or sourceState==NeuronState.Excited
                    if atLeatsOne:
                       self.bufferState = NeuronState.Quiescent
                    else:
                       self.bufferState = NeuronState.Excited
        else:
            self.bufferState = self.state


    # in this model we don't have voltages, we have states but the method needs to be named in this way to maintain
# the same interface with the other models.
    def computeVnextConRs(self, dendriticConnections):
        randomR=np.random.uniform(0.0,1.0,2)
        if not len(dendriticConnections)==0:
            if self.state==NeuronState.Refractary:
                if(randomR[1]<self.testR2):
                    self.bufferState=NeuronState.Quiescent
                else:
                    self.bufferState = NeuronState.Refractary
            elif self.state==NeuronState.Excited:
                self.bufferState=NeuronState.Refractary
            # como otra alternativa aca se podria modelar el caso en que si esta E pero el source es Q con una gap junction entonces bajar la prob de quedarse en R
            elif self.state==NeuronState.Quiescent:
                if((randomR[0]<=self.testR1)):
                    self.bufferState=NeuronState.Excited
                else:
                #if self.neuronName=='REV' and dendriticConnections[0].getSource().getState()==NeuronState.Excited:
                #    print('pasa por REV')
                    sumOfWeights=0
                    for co in dendriticConnections:
                        sourceState=co.getSource().getState()
                        if sourceState==NeuronState.Quiescent:
                            pass
                        if sourceState==NeuronState.Refractary:
                            pass
                        if sourceState==NeuronState.Excited:
                            if (co.isType(con.ConnectionType.ChemEx))or(co.isType(con.ConnectionType.AGJ))or(co.isType(con.ConnectionType.SGJ)):
                                sumOfWeights = sumOfWeights + co.getTestWeight()
                            if co.isType(con.ConnectionType.ChemIn):
                                sumOfWeights = sumOfWeights + co.getTestWeight()
                    if sumOfWeights>self.threshold:
                        self.bufferState=NeuronState.Excited
                    else:
                        self.bufferState=NeuronState.Quiescent
        else:
            self.bufferState=self.state
#prueba sin rs
    def computeVnext(self, dendriticConnections):
        if not len(dendriticConnections)==0:
            if self.state==NeuronState.Refractary:
                    self.bufferState=NeuronState.Quiescent
            elif self.state==NeuronState.Excited:
                self.bufferState=NeuronState.Refractary
            # como otra alternativa aca se podria modelar el caso en que si esta E pero el source es Q con una gap junction entonces bajar la prob de quedarse en R
            elif self.state==NeuronState.Quiescent:
                # if((randomR[0]<=self.testR1)):
                #     self.bufferState=NeuronState.Excited
                # else:
                #if self.neuronName=='REV' and dendriticConnections[0].getSource().getState()==NeuronState.Excited:
                #    print('pasa por REV')
                    sumOfWeights=0
                    for co in dendriticConnections:
                        sourceState=co.getSource().getState()
                        if sourceState==NeuronState.Quiescent:
                            pass
                        if sourceState==NeuronState.Refractary:
                            pass
                        if sourceState==NeuronState.Excited:
                            if (co.isType(con.ConnectionType.ChemEx))or(co.isType(con.ConnectionType.AGJ))or(co.isType(con.ConnectionType.SGJ)):
                                sumOfWeights = sumOfWeights + co.getTestWeight()
                            if co.isType(con.ConnectionType.ChemIn):
                                sumOfWeights = sumOfWeights + co.getTestWeight()
                    if sumOfWeights>self.threshold:
                        self.bufferState=NeuronState.Excited
                    else:
                        self.bufferState=NeuronState.Quiescent
        else:
            self.bufferState=self.state

    #check if still used
    def isSameAs(self,neu2):
        return self.testGleak==neu2.testGleak and self.testVleak==neu2.testVleak and self.testCm==neu2.testCm

#needed for Random Seek
    def commitNoise(self):
        self.threshold = self.testThreshold
        self.r1=self.testR1
        self.r2=self.testR2

    def revertNoise(self):
        self.testThreshold = self.threshold
        self.testR1=self.r1
        self.testR2=self.r2

#needed for PYGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testThreshold,self.testR1,self.testR2]

#needed for HyperOpt
#REVISAR LIMITES
    def getVariablesForHyperOpt(self):
        return {
                #self.neuronName+'_tth':hp.uniform(self.neuronName+'_tth',0.0,1.0),
                self.neuronName+'_tth':hp.uniform(self.neuronName+'_tth',0.0,1.0),
                self.neuronName + '_tr1': hp.uniform(self.neuronName + '_tr1', 0.0, 0.001),
                self.neuronName + '_tr2': hp.uniform(self.neuronName + '_tr2', 0.5, 0.9),
        }






def loadNeuron(xmlNeu):
    neuToReturn = Neuron(xmlNeu.attrib['name'])
    neuToReturn.state=NeuronState.Quiescent
    neuToReturn.threshold=float(xmlNeu.attrib['threshold'])
    neuToReturn.r1=float(xmlNeu.attrib['r1'])
    neuToReturn.r2=float(xmlNeu.attrib['r2'])
    return neuToReturn




class NeuronState(enum.Enum):
    Quiescent = 0
    Refractary = -1
    Excited = 1

    def __str__(self):
        if self==NeuronState.Quiescent:
            return 'Q'
        if self==NeuronState.Refractary:
            return 'R'
        if self==NeuronState.Excited:
            return 'E'

    def __repr__(self):
        return 'mmmmm'

def neuronStateFromStr(str):
    if str=='NeuronState.Quiescent':
        return NeuronState.Quiescent
    if str=='NeuronState.Refractary':
        return NeuronState.Refractary
    if str=='NeuronState.Excited':
        return NeuronState.Excited





