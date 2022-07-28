import matplotlib.cm as cm
import matplotlib.pyplot
import networkx as nx
import Utils.MyNetworkx as mynx
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from matplotlib import colors

import Models.Izhikevich.Connection as conIZ
import Models.IFNeuronalCircuit.Connection as conIF
import Models.Haimovici.Connection as conHA
import Models.Haimovici.Neuron as neuHA
import Models.Fiuri.Connection as conFI
import Models.Fiuri.Neuron as neuFI
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow,Circle,Rectangle



#For TW circuit

def getPotencialNormalizedForColor(version,pot):
    if version=='IZ':
        norm = mplcol.Normalize(vmin=-100.0, vmax=30.0)
        return norm(pot)
    elif version=='IF':
        norm = mplcol.Normalize(vmin=-90.0, vmax=-10.0)
        return norm(pot)
    elif version == 'FI':
        norm = mplcol.Normalize(vmin=-20.0, vmax=-20.0)
        return norm(pot)
    elif version=='HA':
        value=0
        if pot==neuHA.NeuronState.Quiescent:
            value=4
        elif pot==neuHA.NeuronState.Excited:
            value=8
        elif pot==neuHA.NeuronState.Refractary:
            value=6
        norm = mplcol.Normalize(vmin=0.0, vmax=10.0)
        return norm(value)


def stateToShow(version,state):
    toShow=''
    if (version=='IZ' or version=='IF' or version=='FI'):
        toShow=str(round(state,2))
    elif (version=='HA'):
        toShow=str(state)
    return toShow

def graphSnapshot(version,model,index ,extendedLabels=False):
    conTypeEx = None
    conTypeIn = None
    conTypeSGJ = None
    conTypeAGJ = None
    if version == 'HA':
       conTypeEx=conHA.ConnectionType.ChemEx
       conTypeIn=conHA.ConnectionType.ChemIn
       conTypeSGJ=conHA.ConnectionType.SGJ
       conTypeAGJ=conHA.ConnectionType.AGJ
    if version == 'FI':
       conTypeEx=conFI.ConnectionType.ChemEx
       conTypeIn=conFI.ConnectionType.ChemIn
       conTypeSGJ=conFI.ConnectionType.SGJ
       conTypeAGJ=conFI.ConnectionType.AGJ
    if version == 'IF':
       conTypeEx=conIF.ConnectionType.ChemEx
       conTypeIn=conIF.ConnectionType.ChemIn
       conTypeSGJ=conIF.ConnectionType.SGJ
       conTypeAGJ=conIF.ConnectionType.AGJ
    elif version=='IZ':
        conTypeEx = conIZ.ConnectionType.ChemEx
        conTypeIn = conIZ.ConnectionType.ChemIn
        conTypeSGJ = conIZ.ConnectionType.SGJ
        conTypeAGJ = conIZ.ConnectionType.AGJ
    blueCmap =cm.get_cmap('Blues')
    rdPuCmap =cm.get_cmap('RdPu')
    rdPuCmap.set_under('w')
    rdPuCmap.set_over('k')
    oranCmap =cm.get_cmap('Oranges')
    oranCmap.set_over('k')
    oranCmap.set_under('w')
    purCmap =cm.get_cmap('Purples')
    purCmap.set_over('k')
    purCmap.set_under('w')
    greenCmap =cm.get_cmap('Greens')
    greenCmap.set_under('w')
    greenCmap.set_over('k')
    inputsForX =[]
    inputsForV =[]
    outputs =[]
    intermediate =[]
    inputsForXPos ={}
    inputsForVPos ={}
    outputsPos ={}
    intermediatePos ={}
    allPositions ={}
    inputsForXColorMap =[]
    inputsForVColorMap =[]
    outputsColorMap =[]
    intermediateColorMap =[]
    G=nx.DiGraph()
    #----- adding the nodes-------
    nameOfMethodToGetState=''
    if (version=='IZ' or version=='IF'):
        nameOfMethodToGetState='getNeuronPotencialOfName'
    elif (version=='HA'):
        nameOfMethodToGetState='getNeuronStateOfName'
    elif (version=='FI'):
        nameOfMethodToGetState='getInternalStateOfName'
    methodToGetState=getattr(model.neuralnetwork,nameOfMethodToGetState)
    state=''
    state=methodToGetState("FWD")
    col=greenCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("FWD", potencial=stateToShow(version,state))
    outputs.append("FWD")
    outputsPos["FWD"]=(2.0 ,1.5)
    allPositions["FWD"]=(2.0 ,1.5)
    outputsColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("REV"))
    state=methodToGetState("REV")
    col = greenCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("REV",potencial=stateToShow(version,state))
    outputs.append("REV")
    outputsPos["REV"]=(1.0 ,1.5)
    allPositions["REV"]=(1.0 ,1.5)
    outputsColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("PLM"))
    state = methodToGetState("PLM")
    col=rdPuCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("PLM",potencial=stateToShow(version,state))
    inputsForX.append("PLM")
    inputsForXPos["PLM"]=(1, 13 )
    allPositions["PLM"]=(1, 13 )
    inputsForXColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("AVM"))
    state = methodToGetState("AVM")
    col=rdPuCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("AVM",potencial=stateToShow(version,state))
    inputsForX.append("AVM")
    inputsForXPos["AVM"]=(0.5, 8)
    allPositions["AVM"]=(0.5, 8)
    inputsForXColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("PVD"))
    state = methodToGetState("PVD")
    col=oranCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("PVD",potencial=stateToShow(version,state))
    inputsForV.append("PVD")
    inputsForVPos["PVD"]=(7, 8)
    allPositions["PVD"]=(7, 8)
    inputsForVColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("ALM"))
    state = methodToGetState("ALM")
    col=oranCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("ALM",potencial=stateToShow(version,state))
    inputsForV.append("ALM")
    inputsForVPos["ALM"]=(7, 13 )
    allPositions["ALM"]=(7, 13 )
    inputsForVColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("AVA"))
    state = methodToGetState("AVA")
    col=purCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("AVA",potencial=stateToShow(version,state))
    intermediate.append("AVA")
    intermediatePos["AVA"]=(1.0, 5 )
    allPositions["AVA"]=(1.0, 5 )
    intermediateColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("AVD"))
    state = methodToGetState("AVD")
    col=purCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("AVD",potencial=stateToShow(version,state))
    intermediate.append("AVD")
    intermediatePos["AVD"]=(4.5, 9)
    allPositions["AVD"]=(4.5, 9)
    intermediateColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("PVC"))
    state = methodToGetState("PVC")
    col=purCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("PVC",potencial=stateToShow(version,state))
    intermediate.append("PVC")
    intermediatePos["PVC"]=(3, 15 )
    allPositions["PVC"]=(3, 15 )
    intermediateColorMap.append(col)
    #state=(model.neuralnetwork.getNeuronStateOfName("AVB"))
    state = methodToGetState("AVB")
    col=purCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("AVB",potencial=stateToShow(version,state))
    intermediate.append("AVB")
    intermediatePos["AVB"]=(2.75, 7.0 )
    allPositions["AVB"]=(2.75, 7.0 )
    intermediateColorMap.append(col)
    fig=plt.figure()
    fig.suptitle("State resolution step:%s"%(str(index)))
    #state=(model.neuralnetwork.getNeuronStateOfName("DVA"))
    state = methodToGetState("DVA")
    col=purCmap(getPotencialNormalizedForColor(version,state))
    G.add_node("DVA",potencial=stateToShow(version,state))
    intermediate.append("DVA")
    intermediatePos["DVA"]=(6, 4 )
    allPositions["DVA"]=(6, 4 )
    intermediateColorMap.append(col)
    #----- manipulating edges-------
    chemExConList=[co for co in model.neuralnetwork.connections if co.isType(conTypeEx)]
    chemInConList=[co for co in model.neuralnetwork.connections if co.isType(conTypeIn)]
    GJConList=[co for co in model.neuralnetwork.connections if co.isType(conTypeSGJ) or co.isType(conTypeAGJ)]
    edgeColors={}
    for c in (chemExConList):
        G.add_edge(c.connSource.getName(), c.connTarget.getName(), weight=(round(c.connWeight, 2)))
        edgeColors[c.connSource.getName()+c.connTarget.getName()]='#a61107'
    for c in (chemInConList):
        G.add_edge(c.connSource.getName(), c.connTarget.getName(), weight=(round(c.connWeight, 2)))
        edgeColors[c.connSource.getName()+c.connTarget.getName()]='#a6a3b5'
    for c in (GJConList):
        G.add_edge(c.connSource.getName(), c.connTarget.getName(), weight=(round(c.connWeight, 2)))
        edgeColors[c.connSource.getName()+c.connTarget.getName()]='orange'
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    colorListForStraight=[]
    for straight in straight_edges:
        colorListForStraight.append(edgeColors[straight[0]+straight[1]])
    colorListForCurved=[]
    for curved in curved_edges:
        colorListForCurved.append(edgeColors[curved[0]+curved[1]])
    #------------- drawing nodes and edges
    nx.draw_networkx_nodes(G,nodelist=outputs,pos=outputsPos,node_shape='v',node_size=700,node_color=outputsColorMap)
    nx.draw_networkx_nodes(G,nodelist=inputsForX,pos=inputsForXPos,node_shape='>',node_size=700,node_color=inputsForXColorMap)
    nx.draw_networkx_nodes(G,nodelist=inputsForV,pos=inputsForVPos,node_shape='<',node_size=700,node_color=inputsForVColorMap)
    nx.draw_networkx_nodes(G,nodelist=intermediate,pos=intermediatePos,node_shape='o',node_size=700,node_color=intermediateColorMap)

    nx.draw_networkx_edges(G, pos=allPositions, edgelist=straight_edges,edge_color=colorListForStraight,arrows=True,arrowstyle='->',node_size=700)
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos=allPositions, edgelist=curved_edges,edge_color=colorListForCurved,arrows=True,arrowstyle='->',node_size=700,connectionstyle=f'arc3, rad = {arc_rad}')

    #nx.draw_networkx_edges(G,pos=allPositions, edge_color=edgeColors)
    #---------------manipulating edge labels
    #labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos=allPositions, edge_labels=labels)

    edge_weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    mynx.my_draw_networkx_edge_labels(G, pos=allPositions, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad,font_size=8)
    nx.draw_networkx_edge_labels(G, pos=allPositions,  edge_labels=straight_edge_labels, rotate=False,label_pos=0.7,font_size=8)

    #---------------manipulating edge labels
    nodeLabels = {}
    if extendedLabels:
        nodePots=nx.get_node_attributes(G,'potencial')
        for n,p in nodePots.items():
            nodeLabels[n]="%s\n ( %s)"%(n,p)
        # nx.draw_networkx_labels(G, labels=nodeLabels ,pos=allPositions)
        # nx.relabel_nodes(G, nodeLabels)
    # else:

    nx.draw_networkx_labels(G, labels=nodeLabels, pos=allPositions)

    lineHandle=Line2D([0, 1], [0, 1], color='#a61107')
    lineHandle2=Line2D([0, 1], [0, 1], color='#a6a3b5')
    lineHandl3=Line2D([0, 1], [0, 1], color='orange')
    col=greenCmap(getPotencialNormalizedForColor('HA',neuHA.NeuronState.Refractary))
    lineRec=Rectangle((0,0),1,1,color=col,ec="k")
    col=rdPuCmap(getPotencialNormalizedForColor('HA',neuHA.NeuronState.Refractary))
    lineRec2=Rectangle((0,0),1,1,color=col,ec="k")
    col=oranCmap(getPotencialNormalizedForColor('HA',neuHA.NeuronState.Refractary))
    lineRec3=Rectangle((0,0),1,1,color=col,ec="k")
    col=purCmap(getPotencialNormalizedForColor('HA',neuHA.NeuronState.Refractary))
    lineRec4=Rectangle((0,0),1,1,color=col,ec="k")
    #arr=Arrow((0,0),1,1,color='green',ec="k")
    handles1 = [lineRec,lineRec2,lineRec3,lineRec4]
    handles2 = [lineHandle,lineHandle2,lineHandl3]
    labels1 = ['Motor','Sensory(x)','Sensory (v)','Internal']
    labels2=['Exitatory','Inhibitory','Gap Junction']
    legend2=plt.legend(handles2,labels2)
    plt.legend(handles1,labels1,loc='lower center')
    plt.gca().add_artist(legend2)

    # nx.draw_networkx_labels(G, pos=allPositions)
    # nx.relabel_nodes(G, nodeLabels)
# Modificado solo para debug, sacarlo !
    fig.savefig('/tmp/graph%s.png' % (index), bbox_inches='tight')
    #fig.savefig(model.generationGraphModeFolder + '/graph%s.png'%(index ), bbox_inches='tight')
    # plt.show()

def graphReplayStepsHist(optimizer, model, results,folder):
    plt.figure()
    N,val, patches = plt.hist(results,20)

    maxN=np.max(N)

    mean = np.mean(results)
    var = np.var(results, ddof=0)
    std = np.std(results, ddof=0)
    stepsMedian = np.median(results)
    maxVal = max(results)
    minVal = min(results)

    cmap = plt.get_cmap('jet')
    low = cmap(0.8)
    medium = cmap(0.6)
    medium2 =cmap(0.4)
    high = cmap(0.25)

    for i in range(0,len(patches)):
        if val[i]==minVal:
            patches[i].set_facecolor(low)
        elif val[i]<mean:
            patches[i].set_facecolor(medium2)
        elif val[i]>999.0-((maxVal-minVal)/20.0)-1:
            patches[i].set_facecolor(high)
        else:
            patches[i].set_facecolor(medium)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium,medium2, high] ]
    handles.append(Rectangle((0,0),1,1,color='red',ec="k"))
    labels= ["Min: %s"%(minVal),"More than mean", "Less than mean", "Unsuccessful","Mean:"+str(np.round(mean,2))]
    plt.legend(handles, labels)
    plt.title('%s with %s: steps histogram'%(model,optimizer))
    plt.xlabel("Environment interactions", fontsize=12)
    plt.ylabel("Ran ammount", fontsize=12)
    plt.xticks(fontsize=11)
    #plt.xticks(rotation=45)
    plt.yticks(fontsize=11)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(True)
    plt.vlines(mean, 0, maxN, color='red', linestyles='solid', label=str(mean))
    strText=r'$\mu=$' + str(round(mean,2)) + '\n $\sigma=$' + str(round(std,2)) + '\n $\sigma^2$=' + str(round(var, 2))+'\n med='+str(round(stepsMedian,2))
    textXpos = minVal + (maxVal - minVal) / 2
    textYpos = 2*maxN / 3
    plt.text(textXpos, textYpos,strText , horizontalalignment='center',verticalalignment='top')
    plt.savefig(folder + '/hist2.png')
    plt.show()


def graphReplayRewardsHist(optimizer, model, results,folder):
    plt.figure()
    N,val, patches = plt.hist(results,20)

    maxN=np.max(N)
    mean = np.mean(results)
    var = np.var(results, ddof=0)
    std = np.std(results, ddof=0)

    maxVal = max(results)
    minVal = min(results)

    cmap = plt.get_cmap('jet')
    low = cmap(0.8)
    medium = cmap(0.25)
    high = cmap(0.5)

    for i in range(0,len(patches)):
        if val[i]<0:
            patches[i].set_facecolor(low)
        elif val[i]>91.5:
            patches[i].set_facecolor(high)
        else:
            patches[i].set_facecolor(medium)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high] ]
    handles.append(Rectangle((0,0),1,1,color='red',ec="k"))
    labels= ["Unsuccessful","Successful <= 91.5", "Successful > 91.5", "Mean: "+str(np.round(mean,2))]
    plt.legend(handles, labels)
    plt.title('%s with %s: rewards histogram'%(model,optimizer))
    plt.xlabel("Rewards", fontsize=12)
    plt.ylabel("Run ammount", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(True)
    plt.vlines(mean, 0, maxN, color='red', linestyles='solid', label=str(mean))
    strText=r'$\mu=$' + str(round(mean,2)) + '\n $\sigma=$' + str(round(std,2)) + '\n $\sigma^2$=' + str(round(var, 2))
    textXpos = minVal + (maxVal - minVal) / 2
    textYpos = 4*maxN / 5
    plt.text(textXpos, textYpos,strText , horizontalalignment='center',verticalalignment='top')
    plt.savefig(folder + '/hist.png')
    plt.show()

def graphVarTraces(trace,folder,who):
    print(who)
    print (folder)
    print (trace)


def graphComparissonRew (xs,ys,err,folder):

    fig, ax = plt.subplots()
    plt.title('Configuration comparisson')
    plt.xlabel('Configuration')
    plt.ylabel('Reward')

    ax.errorbar(xs, ys, err, fmt='o', linewidth=2, capsize=6 ,color='green')

    plt.savefig(folder+ 'rewComparisson.png', bbox_inches='tight')
    plt.show()
    #
def cm_to_inch(value):
    return value/2.54

def  graphComparissonSteps(xs,ys,err,mins,folder):
    fig, ax = plt.subplots()
    fig.set_figwidth(cm_to_inch(30))
    handles = [Rectangle((0,0),1,1,color='red',ec="k"),Rectangle((0,0),1,1,color='orange',ec="k")]
    labels = ['Min','Mean']
    ax.errorbar(xs, ys, err, fmt='o', linewidth=2, capsize=6 ,color='orange')
    plt.xlabel('Configuration')
    plt.xticks(fontsize=9)
    plt.xticks(rotation=90)
    #ticksCols=['red','red','red','red','red','blue','blue','blue','blue','blue','red','red','red','red','red','blue', 'blue', 'blue', 'blue', 'blue','red','red','red','red','red']
    ax.tick_params(axis='x', width=12)
    plt.tick_params(axis='x',colors='red')
    plt.ylabel('Steps')
    plt.title('Configuration comparisson')
    plt.plot(xs, mins, 'ro', label='reward')
    plt.legend(handles,labels)
    plt.savefig(folder+ '/stepsComparisson.png', bbox_inches='tight')
    plt.show()


