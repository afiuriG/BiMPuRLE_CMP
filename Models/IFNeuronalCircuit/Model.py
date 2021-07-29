import Models.IFNeuronalCircuit.NeuralNetwork as nN
from Models import IFNeuronalCircuit as de
import Models.IFNeuronalCircuit.ModelInterfaces as mi
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom


#aca iran los AddNoise, el update y capaz que aLGUNA OTRA COSA
np.random.seed(423)
neuronRandomIndexedNames = []



class Model:
    def __init__(self,name):
        self.name = name
        self.neuralnetwork = None
        self.interfaces = {}
    def setRandomIndexesList(self,list):
        neuronRandomIndexedNames=list
    def loadModelHarcodded(self):
        self.neuralnetwork = self.loadNN()
        self.interfaces = self.loadInterfaces()

    def load(self,fromFile=''):
        if fromFile=='':
            self.loadModelHarcodded()
        else:
            self.loadFromFile(fromFile)

    def __str__(self):
        return """
        RED:
        %s
        INTERFACES:
        %s
        """ % (self.neuralnetwork," ".join(str(i) for i in self.interfaces.values()))
    def __repr__(self):
        return """RED:
            %s
        INTERFACES:
            %s
            """ % (self.neuralnetwork, " ".join(str(i) for i in self.interfaces))

    def loadNN(self):
        twnn = nN.NeuralNetwork('TW_I&F_Letch')
        twnn.loadNeuron('PVD')
        neuronRandomIndexedNames.append('PVD')
        twnn.loadNeuron('PLM')
        neuronRandomIndexedNames.append('PLM')
        twnn.loadNeuron('AVA')
        neuronRandomIndexedNames.append('AVA')
        twnn.loadNeuron('AVD')
        neuronRandomIndexedNames.append('AVD')
        twnn.loadNeuron('PVC')
        neuronRandomIndexedNames.append('PVC')
        twnn.loadNeuron('AVB')
        neuronRandomIndexedNames.append('AVB')
        twnn.loadNeuron('AVM')
        neuronRandomIndexedNames.append('AVM')
        twnn.loadNeuron('ALM')
        neuronRandomIndexedNames.append('ALM')
        twnn.loadNeuron('DVA')
        neuronRandomIndexedNames.append('DVA')
        twnn.loadNeuron('FWD')
        neuronRandomIndexedNames.append('FWD')
        twnn.loadNeuron('REV')
        neuronRandomIndexedNames.append('REV')
        #cargar el resto de las neuronas....
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PVD', 'AVA', 0.60996)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PLM', 'AVA', 0.84907)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'PVC', 'AVA', 1.327515)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVB', 'AVA', 0.22908)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'AVD', 'AVA', 1.656176)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PLM', 'AVD', 0)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'PVC', 'AVD', 0.900241)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVA', 'AVD', 0.488061)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVB', 'AVD', 0.443871)
        twnn.loadConnection(de.ConnectionType.AGJ, 'AVM', 'AVD', 1.72995)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'ALM', 'AVD', 0)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PVD', 'PVC', 0.59057)
        twnn.loadConnection(de.ConnectionType.AGJ, 'PLM', 'PVC', 0.0388042)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVA', 'PVC', 0.2)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'AVD', 'PVC', 1.343)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVM', 'PVC', 0.147041)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'ALM', 'PVC', 0.70817)
        #esta puede estar mal y ser al reves
        twnn.loadConnection(de.ConnectionType.ChemIn, 'DVA', 'PVC', 0.865243)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'PVC', 'AVB', 0.948166)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'AVA', 'AVB', 0.58725)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'AVD', 'AVB', 1.53322)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PVD', 'DVA', 0.90098)
        twnn.loadConnection(de.ConnectionType.ChemIn, 'PLM', 'DVA', 0.91319)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'PVC', 'DVA', 1.427991)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'AVB', 'FWD', 2.0)
        twnn.loadConnection(de.ConnectionType.ChemEx, 'AVA', 'REV', 2.0)
        #cargar el resto de las conexiones....
        return twnn



    #FIJARE BIEN CUAL ES CUAL DE LAS DE ENTRADA Y SALIDA
    # AddBiSensoryNeuron(1,6,-0.3,0.3)
    # AddBiSensoryNeuron(7,0,-0.02,0.02)
    # AddBiMotorNeuron(9,10,-1,1)
    def loadInterfaces(self):
        interfaces = {}
        twnn = self.neuralnetwork
        nPLM = twnn.getNeuron('PLM')
        nAVM = twnn.getNeuron('AVM')
        inputInterface = mi.BinaryInterface('IN1', nPLM, nAVM, 'IN')
        inputInterface.setMinValue(-0.3)
        inputInterface.setMaxValue(0.3)
        interfaces['IN1'] = inputInterface
        nALM = twnn.getNeuron('ALM')
        nPVD = twnn.getNeuron('PVD')
        inputInterface = mi.BinaryInterface('IN2', nALM, nPVD, 'IN')
        inputInterface.setMinValue(-0.02)
        inputInterface.setMaxValue(0.02)
        interfaces['IN2'] = inputInterface
        nFWD = twnn.getNeuron('FWD')
        nREV = twnn.getNeuron('REV')
        inputInterface = mi.BinaryInterface('OUT1',nFWD, nREV, 'OUT')
        inputInterface.setMinValue(-1)
        inputInterface.setMaxValue(1)
        interfaces['OUT1'] = inputInterface
        return interfaces

    def getInterface(self,name):
        return self.interfaces[name]

    def getNeuron(self,name):
        return self.neuralnetwork.getNeuron(name)

    def getName(self):
        return self.name
#Reset
    def Reset(self):
        self.neuralnetwork.resetAllNeurons()
        for intf in self.interfaces.values():
            intf.reset()




    def DumpClear(self):
        print('se ejecuto DumpClear')


#Update(observations,0.01,10)
#Update(observations,deltaT,simmulationSteps):
#mas que un update esto es un runConnectome ya que toma las entradas (observaciones)
# y corre el connectoma con estas entradas a un paso de deltaT por simmulationSteps pasos
#se mantiene el nombre mientras se use el mismo codigo del autor del paper
    def Update(self,observations,deltaT,simmulationSteps,doLog=False):
        if doLog:
            networkLog = open('log/oneEpisodeModel.log', 'a')
            networkLog.write("----Update(%s,%s,%s)" % (observations,deltaT,simmulationSteps) + '\n')
            networkLog.write('\t' + "Pre state" + '\n')
        for step in range(0,simmulationSteps-1):
            #seteo valores de las entradas
            self.interfaces['IN1'].setValue(observations[0])
            self.interfaces['IN2'].setValue(observations[1])
            #transformo y cargo las variables x y v a la red
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            # hago DoSimulationStep(deltaT)
            if doLog:
                self.neuralnetwork.writeToFile(networkLog)
            self.neuralnetwork.doSimulationStep(deltaT)
            if doLog:
                self.neuralnetwork.writeToFile(networkLog)
                networkLog.write('\t' + "------------------------------------" + '\n')
                networkLog.flush()
            # recupero el valor de output
            ret = self.interfaces['OUT1'].getFeedBackNN()

            #return el valor de output
            return ret

    # CommitNoise()
    def commitNoise(self):
        #se comitea sobre todas, capaz que para hacerlo mas rapido hay que comitear sobre las que fueron modif nomas
        self.neuralnetwork.commitNoise()

    # UndoNoise
    def revertNoise(self):
        #se comitea sobre todas, capaz que para hacerlo mas rapido hay que comitear sobre las que fueron modif nomas
        self.neuralnetwork.revertNoise()

    # AddNoise(0.5, 15)
    # self.lif.AddNoiseVleak(8, 8)
    # self.lif.AddNoiseGleak(0.2, 8)
    # self.lif.AddNoiseSigma(0.2, 10)
    # self.lif.AddNoiseCm(0.1, 10)
    # self.lif.CommitNoise()

    def addNoise(self,parameter,varianza,samplesAmmount):
        mu, sigma = 0, varianza
        normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
        # esta distrib puede repetir valores, ej probar con np.random.randint(0,10,8) y la semilla que ya esta puesta
        if parameter == 'Gleak' or parameter == 'Vleak' or parameter == 'Cm' :
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
        elif parameter == 'Sigma' or parameter == 'Weight':
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
        #print("----------- en addNoise(%s, %s, %s)" % (parameter, varianza, samplesAmmount))
        #print("--------------- se ejecuta: noiseParam(%s,%s,%s,%s)" % (samplesAmmount,parameter,uniformDistribuitedValues,normalDistribuitedValues))
        self.noiseParam(samplesAmmount,parameter,uniformDistribuitedValues,normalDistribuitedValues)


    def noiseParam(self, samplesAmmount, parameter, whiches, values):
        for i in range(0, samplesAmmount):
            which = whiches[i]
            x = values[i]
            if parameter == 'Weight':
                #print("conexion indice %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                newValue = self.neuralnetwork.getConnectionTestWeightOfIdx(which) + x
                if newValue < 0:
                    newValue = 0
                elif newValue > 3.0:
                    newValue = 3.0
                self.neuralnetwork.setConnectionTestWeightOfIdx(which, newValue)
                #print("conexion indice %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'Sigma':
                #print("conexion indice %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                newValue = self.neuralnetwork.getConnectionTestSigmaOfIdx(which) + x
                if newValue < 0.05:
                    newValue = 0.05
                elif newValue > 0.5:
                    newValue = 0.5
                self.neuralnetwork.setConnectionTestSigmaOfIdx(which, newValue)
                #print("conexion indice %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'Gleak':
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestGleakOfName(neuronRandomIndexedNames[which]) + x
                if newValue < 0.05:
                    newValue = 0.05
                elif newValue > 5.0:
                    newValue = 5.0
                self.neuralnetwork.setNeuronTestGleakOfName(neuronRandomIndexedNames[which],newValue)
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'Vleak':
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestVleakOfName(neuronRandomIndexedNames[which]) + x
                if newValue < -90:
                    newValue = -90
                elif newValue > 0:
                    newValue = 0
                self.neuralnetwork.setNeuronTestVleakOfName(neuronRandomIndexedNames[which], newValue)
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'Cm':
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                newValue = self.neuralnetwork.getNeuronTestCmOfName(neuronRandomIndexedNames[which]) + x
                if newValue < 0.001:
                    newValue = 0.001
                elif newValue > 1.0:
                    newValue = 1.0
                self.neuralnetwork.setNeuronTestCmOfName(neuronRandomIndexedNames[which], newValue)
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))


    def writeToFile(self,logfile):
        logfile.write("NEURAL NETWORK" + '\n')
        self.neuralnetwork.writeToFile(logfile)
        logfile.write("INTERFACES" + '\n')
        for inter in self.interfaces.values():
            logfile.write(str(inter) + '\n')

    def dumpModel(self,dumpFile):
        xmlModel = ET.Element('Model')
        xmlModel.set('name',self.name)
        xmlNn = ET.SubElement(xmlModel, 'NeuralNetwork')
        xmlInterfaces = ET.SubElement(xmlModel, 'Interfaces')
        self.neuralnetwork.dumpNeuralNetwork(xmlNn)
        for inter in self.interfaces.values():
            inter.dumpInterface(xmlInterfaces)
        mydata = minidom.parseString(ET.tostring(xmlModel)).toprettyxml(indent="   ")
        #mydata = ET.tostring(xmlModel)
        myfile = open(dumpFile, "w")
        #myfile.write(str(mydata,'utf-8'))
        myfile.write(mydata)
        myfile.close()

    def isSameAs(self,model2):
        return self.neuralnetwork.isSameAs(model2.neuralnetwork)

    def clone(self):
        modelToRet=Model('COPY')
        modelToRet.neuralnetwork=self.neuralnetwork.clone()
        for key,val in self.interfaces.items():
            modelToRet.interfaces[key]=(self.interfaces[key]).clone()
        return modelToRet


    def loadFromFile(self,fromFile):
        #neuronRandomIndexedNames=[]
        mydoc = ET.parse(fromFile)
        xmlRootModelElement = mydoc.getroot()
        for child in xmlRootModelElement:
            if child.tag == 'NeuralNetwork':
               loaded =  nN.loadNeuralNetwork(child)
               self.neuralnetwork = loaded
               #neuronRandomIndexedNames=loaded[1]
               for n in self.neuralnetwork.getNeurons():
                   neuronRandomIndexedNames.append(n.getName())
            if child.tag == 'Interfaces':
               inter = mi.loadConnection(child,self.neuralnetwork)
               self.interfaces=inter


    def getModelInterface(self):
        return mi

    def getNeuronRandomIndexedNames(self):
        global neuronRandomIndexedNames
        return neuronRandomIndexedNames





