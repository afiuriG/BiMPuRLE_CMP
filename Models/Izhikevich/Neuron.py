
import math
from hyperopt import hp
import Models.Izhikevich.Connection as conn


#k1 = 0.04
k1=0.1
k2 = 5
k3 = 140
Sigmoid_mu = 30
V_peak = 30

class Neuron:


    def __init__(self, nname):
        self.neuronName = nname
        self.paramA = 0.0
        self.paramB = 0.0
        self.paramC = 0.0
        self.paramD = 0.0
        self.potencialV = 0.0
        self.recoveryU = 0.0
        self.testParamA = 0.0
        self.testParamB = 0.0
        self.testParamC = 0.0
        self.testParamD = 0.0
        self.E_inhib = -90
        self.E_exit = 0
        self.fired = False
        self.lastActivationTrace=[]

    def __str__(self):
        return "[%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s]\n" % (
            self.neuronName, self.potencialV, self.recoveryU,self.testParamA, self.testParamB,self.testParamC,self.testParamD,self.paramA, self.paramB,self.paramC,self.paramD)


    def __eq__(self, other):
        if not isinstance(other, Neuron):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.neuronName == other.neuronName

    def getName(self):
        return self.neuronName

    # initialize in rest state
    def initialize(self, a, b, c,d):
        self.paramA = a
        self.paramB = b
        self.paramC = c
        self.paramD = d
        self.potencialV = c
        self.recoveryU = self.paramB*self.paramC
        self.testParamA = a
        self.testParamB = b
        self.testParamC = c
        self.testParamD = d
        #self.E_exit = self.paramC

    def resetLastActivationTrace(self):
        self.lastActivationTrace=[]
    def getLastActivaionTrace(self):
        return self.lastActivationTrace
    def recordActivationTrace(self):
        self.lastActivationTrace.append(self.potencialV)

#state seters and geters
    def setPotencial(self, pot):
        self.potencialV = pot
    def getPotencial(self):
        return self.potencialV
    def setTestPotencial(self, pot):
        self.testPotencialV = pot
    def getTestPotencial(self):
        return self.testPotencialV

    def setRecovery(self,rec):
        self.recoveryU=rec
    def getRecovery(self):
        return self.recoveryU
    def setTestRecovery(self,rec):
        self.testRecoveryU=rec
    def getTestRecovery(self):
        return self.testRecoveryU

    def setFired(self):
        self.fired=True
    def getFired(self):
        return self.fired
    def resetFired(self):
        self.fired=False

#setters and getters for params

    def setParamA(self, a):
        self.paramA = a
    def getParamA(self):
        return self.paramA
    def setParamB(self, b):
        self.paramB = b
    def getParamB(self):
        return self.paramB
    def setParamC(self, c):
        self.paramC = c
    def getParamC(self):
        return self.paramC
    def setParamD(self, d):
        self.paramD = d
    def getParamD(self):
        return self.paramD

    def setTestParamA(self, a):
        self.testParamA = a
    def getTestParamA(self):
        return self.testParamA
    def setTestParamB(self, b):
        self.testParamB = b
    def getTestParamB(self):
        return self.testParamB
    def setTestParamC(self, c):
        self.testParamC = c
    def getTestParamC(self):
        return self.testParamC
    def setTestParamD(self, d):
        self.testParamD = d
    def getTestParamD(self):
        return self.testParamD





    def resetPotencial(self):
        self.potencialV = self.paramC
    def resetRecovery(self):
        self.recoveryU = self.paramB*self.paramC


    def computeV(self, delta, dendriticConnections):
        #calculate the total I for the entry
        v=self.potencialV
        u=self.recoveryU
        tau=delta
        f_en_v_de_t_y_U_de_t=k1*v*v+k2*v+k3-u
        iGJ=0
        iEx = 0
        iIn = 0
        #g_en_t=0
        #sum_gi_Ei=0
        for dc in dendriticConnections:
            sourcePot = dc.getSource().getPotencial()
            if dc.isType(conn.ConnectionType.ChemEx):
                if sourcePot > 0:
                    currI = dc.getTestWeight() * (self.E_exit + sourcePot)
                else:
                    currI = dc.getTestWeight() * (self.E_exit - sourcePot)
                iEx = iEx + currI
            elif dc.isType(conn.ConnectionType.ChemIn):
                currI = dc.getTestWeight() * (self.E_inhib - sourcePot)
                iIn = iIn + currI
            else:
                currI = dc.getTestWeight() * (sourcePot - v)
                iGJ = iGJ + currI
        v_en_t_mas_tau = (v + tau * (f_en_v_de_t_y_U_de_t + iEx + iIn + iGJ))
        if (v_en_t_mas_tau)<-100:
            v_en_t_mas_tau=-100
        if(v_en_t_mas_tau)>V_peak:
            self.fired=True
            tau_peack=(V_peak-v)*tau/(v_en_t_mas_tau-v)
            u_en_t_mas_tau = u + tau_peack * (self.testParamA * (self.testParamB * v - u))
            v_en_t_mas_tau = self.testParamC
            #v_en_t_mas_tau = -20
        else:
            u_en_t_mas_tau = u + tau * (self.testParamA*(self.testParamB*v-u))
        self.potencialV = v_en_t_mas_tau
        self.recoveryU = u_en_t_mas_tau


    def sigmoid(self,sourcePot, sigma):
        try:
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        except OverflowError:
            print('source: %s'%(sourcePot))
            print('sigma: %s'%(sigma))
            value = 1 / (1+math.exp(-sigma*(sourcePot-Sigmoid_mu)))
        return value


#dismissed
    def isSameAs(self,neu2):
        return self.paramA==neu2.paramA and self.paramB==neu2.paramB and self.paramC==neu2.paramC and self.paramD==neu2.paramD and self.testParamA==neu2.testParamA and self.testParamB==neu2.testParamB and self.testParamC==neu2.testParamC and self.testParamD==neu2.testParamD

#needed for Random Seek
    def commitNoise(self):
         self.paramA = self.testParamA
         self.paramB = self.testParamB
         self.paramC = self.testParamC
         self.paramD = self.testParamD

    def revertNoise(self):
         self.testParamA = self.paramA
         self.testParamB = self.paramB
         self.testParamC = self.paramC
         self.testParamD = self.paramD


#needed for PYGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.paramA,self.paramB,self.paramC,self.paramD]

#needed for HyperOpt

    def getVariablesForHyperOpt(self):
        return {
                self.neuronName+'_ta':hp.uniform(self.neuronName+'_ta',0.019999,0.020001),
                self.neuronName + '_tb': hp.uniform(self.neuronName + '_tb', 0.19999, 0.20001),
                self.neuronName + '_tc': hp.uniform(self.neuronName + '_tc', -21.0,-20.0 ),
                self.neuronName + '_td': hp.uniform(self.neuronName + '_td', 7.0, 8.0)
        }





def loadNeuron(xmlNeu):
    neuToReturn = Neuron(xmlNeu.attrib['name'])
    neuToReturn.paramA=float(xmlNeu.attrib['a'])
    neuToReturn.paramB=float(xmlNeu.attrib['b'])
    neuToReturn.paramC=float(xmlNeu.attrib['c'])
    neuToReturn.paramD=float(xmlNeu.attrib['d'])
    neuToReturn.potencialV=float(xmlNeu.attrib['potencialV'])
    neuToReturn.recoveryU=float(xmlNeu.attrib['recoveryU'])
    neuToReturn.E_exit = neuToReturn.paramC
    neuToReturn.revertNoise()
    return neuToReturn




