import Models.IFNeuronalCircuit.ModelInterfaces as modIF
import random



global availableConnectionsToGet
global specialNeurons

# this methods are for shuffling
def shuffle(model):
    global  availableConnectionsToGet
    global specialNeurons
    availableConnectionsToGet=[]
    specialNeurons=[]
    nn=model.neuralnetwork
    areAllConnected =False
    neuronList =list(nn.getNeurons())
#    while not areAllConnected:
    for con in nn.getConnections():
       newSource =random.choice(neuronList)
       newTarget =random.choice(neuronList)
       #no cicles
       while newSource==newTarget:
            newTarget=random.choice(neuronList)
       con.setSource(newSource)
       con.setTarget(newTarget)
       availableConnectionsToGet.append(con)
    normalizeConectivityConditions(model)



def normalizeConectivityConditions(model):
    normalizeSensorial(model)
    normalizeMotors(model)
    normalizeInternals(model)

def normalizeSensorial(model):
    global  availableConnectionsToGet
    global specialNeurons
    inputInterfacesList=[mi for mi in model.getInterfaces() if mi.type == 'IN']
    for sensMI in inputInterfacesList:
        #is instance will check for different interfaces besides the binnary used in TWC
        if isinstance(sensMI,modIF.BinaryInterface):
            posNeu=sensMI.getPosNeu()
            specialNeurons.append(posNeu.getName())
            negNeu=sensMI.getNegNeu()
            specialNeurons.append(negNeu.getName())
            #positive neuron has to be source in at least one conn
            connectionsWithPosNeuAsSource = [con for con in model.neuralnetwork.getConnections() if con.getSource()==posNeu]
            if len(connectionsWithPosNeuAsSource)==0:
                anyOneElseConn=random.choice(availableConnectionsToGet)
                anyOneElseConn.setSource(posNeu)
                if anyOneElseConn in availableConnectionsToGet:
                    availableConnectionsToGet.remove(anyOneElseConn)
            elif len(connectionsWithPosNeuAsSource)==1 and connectionsWithPosNeuAsSource:
                availableConnectionsToGet.remove(connectionsWithPosNeuAsSource[0])
            #negative neuron has to be source in at least one conn
            connectionsWithNegNeuAsSource = [con for con in model.neuralnetwork.getConnections() if con.getSource()==negNeu]
            if len(connectionsWithNegNeuAsSource)==0:
                anyOneElseConn=random.choice(availableConnectionsToGet)
                anyOneElseConn.setSource(negNeu)
                if anyOneElseConn in availableConnectionsToGet:
                    availableConnectionsToGet.remove(anyOneElseConn)
            elif len(connectionsWithNegNeuAsSource)==1 and connectionsWithNegNeuAsSource in availableConnectionsToGet:
                availableConnectionsToGet.remove(connectionsWithNegNeuAsSource[0])

def normalizeMotors(model):
    outputInterfacesList=[mi for mi in model.getInterfaces() if mi.type == 'OUT']
    for motorMI in outputInterfacesList:
        #is instance will check for different interfaces besides the binnary used in TWC
        if isinstance(motorMI,modIF.BinaryInterface):
            posNeu=motorMI.getPosNeu()
            specialNeurons.append(posNeu.getName())
            negNeu=motorMI.getNegNeu()
            specialNeurons.append(negNeu.getName())
            #positive neuron has to be source in at least one conn
            connectionsWithPosNeuAsTarget = [con for con in model.neuralnetwork.getConnections() if con.getTarget()==posNeu]
            if len(connectionsWithPosNeuAsTarget)==0:
                anyOneElseConn=random.choice(availableConnectionsToGet)
                anyOneElseConn.setSource(posNeu)
                if anyOneElseConn in availableConnectionsToGet:
                    availableConnectionsToGet.remove(anyOneElseConn)
            elif len(connectionsWithPosNeuAsTarget)==1 and connectionsWithPosNeuAsTarget in availableConnectionsToGet:
                availableConnectionsToGet.remove(connectionsWithPosNeuAsTarget[0])
            #negative neuron has to be source in at least one conn
            connectionsWithNegNeuAsTarget = [con for con in model.neuralnetwork.getConnections() if con.getTarget()==negNeu]
            if len(connectionsWithNegNeuAsTarget)==0:
                anyOneElseConn=random.choice(availableConnectionsToGet)
                anyOneElseConn.setSource(negNeu)
                if anyOneElseConn in availableConnectionsToGet:
                    availableConnectionsToGet.remove(anyOneElseConn)
            elif len(connectionsWithNegNeuAsTarget)==1 and connectionsWithNegNeuAsTarget in availableConnectionsToGet:
                availableConnectionsToGet.remove(connectionsWithNegNeuAsTarget[0])

def normalizeInternals(model):
    #internals should be at least once source and at least once target
    intNeuList=[neu for neu in model.neuralnetwork.getNeurons() if neu.getName() not in specialNeurons]
    for intNeu in intNeuList:
        connectionsWithNeuAsSource = [con for con in model.neuralnetwork.getConnections() if con.getSource() == intNeu]
        if len(connectionsWithNeuAsSource) == 0:
            anyOneElseConn = random.choice(availableConnectionsToGet)
            anyOneElseConn.setSource(intNeu)
            if anyOneElseConn in availableConnectionsToGet:
                availableConnectionsToGet.remove(anyOneElseConn)
        elif len(connectionsWithNeuAsSource) == 1 and connectionsWithNeuAsSource[0] in availableConnectionsToGet:
            availableConnectionsToGet.remove(connectionsWithNeuAsSource[0])
        connectionsWithNeuAsTarget = [con for con in model.neuralnetwork.getConnections() if con.getTarget() == intNeu]
        if len(connectionsWithNeuAsTarget) == 0:
            anyOneElseConn = random.choice(availableConnectionsToGet)
            anyOneElseConn.setSource(intNeu)
            if anyOneElseConn in availableConnectionsToGet:
                availableConnectionsToGet.remove(anyOneElseConn)
        elif len(connectionsWithNeuAsTarget) == 1 and connectionsWithNeuAsTarget[0] in availableConnectionsToGet:
             availableConnectionsToGet.remove(connectionsWithNeuAsTarget[0])

# End shuffling methods

