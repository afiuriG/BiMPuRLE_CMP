

import csv
from datetime import  timedelta
import Utils.GraphUtils as gra


global results
global scenariosSufix
global models
global optimSteps
global optimizers
global pulseStats

global bestValuesByModelStats
global bestRewByModel
global bestStepByModel
global bestMinStepByModel


def generateAllGraphs(rootPath):
    parseAllDatas(rootPath)
    statsGenForComparisson()
    generateGraphs()
    generatePulseGraph()
    generateComparissonsGraphs()
    #geneatePonderatedBasedGraphs()


def generatePulseGraph():
    global pulseStats
    configsLab=['FIU','HAI','IZH','IandF']
    values=[]
    linelabels=[]
    for k,v in pulseStats.items():
        values.append((pulseStats[k])[0])
        linelabels.append((pulseStats[k])[1])
    gra.graphPulseComparisson(configsLab,values,linelabels)


def generateGraphs():
    global models
    for mod in models:
        generateModelBasedGraph(mod)


def generateModelBasedGraph(modelShortName):
    for opt in optimizers:
        generateOptStepsModelBasedGraph(modelShortName,opt,'Rewards')
        generateOptStepsModelBasedGraph(modelShortName,opt,'Steps')
        generateTimesOptStepsModelBasedGraph(modelShortName, opt)


def generateOptStepsModelBasedGraph(modelShortName,optShortName,basedOn):
    global scenariosSufix
    global optimSteps
    global results
#    optSt=[x for x in optimSteps if optShortName in x]
    params=[]
    params2=[]
    configsLab = scenariosSufix
    referenceLabels=[]
    for opst in [x for x in optimSteps if optShortName in x]:
        #opst es por ejemplo GA20 y hay que armar las series para el graf de  IZHGA20
        #armar las series
        #graficar
        means = []
        sigmas = []
        points = []
        referenceLabels.append(opst.replace(optShortName,""))
        datasToProcess = getResultsKeysWithCondition(modelShortName+opst)
        for dataKey in datasToProcess:
            data=results[dataKey]
            if data=={}:
                means.append(0.0)
                sigmas.append(0.0)
                points.append(0.0)
            else:
                if basedOn=='Rewards':
                    means.append(data['replayMeanRew'])
                    sigmas.append(data['replaySigmaRew'])
                    points.append(data['trainRew'])
                    titLabel = 'evaluation rewards'
                elif basedOn=='Steps':
                    means.append(data['replayMeanStep'])
                    sigmas.append(data['replaySigmaStep'])
                    points.append(data['replayMinStep'])
                    titLabel = 'evaluation steps'
        params.append(means)
        params.append(sigmas)
        params2.append(points)
        if basedOn=='Rewards':
            version='reward'
            sugar='train'
            pointtitLabel = 'train reward'
        elif basedOn=='Steps':
            version='step'
            sugar='min'
            pointtitLabel='min of steps'
    gra.graphConfigurationComparisson(configsLab,referenceLabels,basedOn,modelShortName+optShortName,params,'replay',version,titLabel)
    gra.graphConfigurationComparisson(configsLab,referenceLabels,basedOn,modelShortName+optShortName,params2,sugar,version,pointtitLabel)

def generateComparissonsGraphs():
    global bestValuesByModelStats
    xLab=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    models=['IandF','IZH','FIU']
    rewParams={}
    stepParams={}
    minStepParams={}
    for model in models:

        #generating series for Mean Reward based comparission
        alist=bestValuesByModelStats[model]['MeanRew']
        rewConfigLabels={'RS':[],'GA':[],'BO':[]}
        rewYsArray ={'RS':[],'GA':[],'BO':[]}
        rewYsErrArray ={'RS':[],'GA':[],'BO':[]}
        rewPositions = {'RS':[],'GA':[],'BO':[]}
        index=0
        counter=0
        while counter<15:
            if((alist[index])[1]['replaySigmaRew']<2):
                if 'RS' in (alist[index])[0]:
                    rewYsArray['RS'].append((alist[index])[1]['replayMeanRew'])
                    rewYsErrArray['RS'].append((alist[index])[1]['replaySigmaRew'])
                    rewConfigLabels['RS'].append((alist[index])[0])
                    counter=counter+1
                    rewPositions['RS'].append(counter)
                elif 'GA' in (alist[index])[0]:
                    rewYsArray['GA'].append((alist[index])[1]['replayMeanRew'])
                    rewYsErrArray['GA'].append((alist[index])[1]['replaySigmaRew'])
                    rewConfigLabels['GA'].append((alist[index])[0])
                    counter=counter+1
                    rewPositions['GA'].append(counter)
                elif 'BO' in (alist[index])[0]:
                    rewYsArray['BO'].append((alist[index])[1]['replayMeanRew'])
                    rewYsErrArray['BO'].append((alist[index])[1]['replaySigmaRew'])
                    rewConfigLabels['BO'].append((alist[index])[0])
                    counter = counter + 1
                    rewPositions['BO'].append(counter)
            index=index+1
        rewParams[model]=rewYsArray,rewYsErrArray,rewConfigLabels,rewPositions

        #generating series for Mean Steps based comparission
        alist=bestValuesByModelStats[model]['MeanStep']
        stepConfigLabels={'RS': [], 'GA': [], 'BO': []}
        stepYsArray={'RS': [], 'GA': [], 'BO': []}
        stepYsErrArray={'RS': [], 'GA': [], 'BO': []}
        stepPositions = {'RS': [], 'GA': [], 'BO': []}
        index=0
        counter=0
        while counter<15:
            if((alist[index])[1]['replaySigmaStep']<100):
                if 'RS' in (alist[index])[0]:
                    stepYsArray['RS'].append((alist[index])[1]['replayMeanStep'])
                    stepYsErrArray['RS'].append((alist[index])[1]['replaySigmaStep'])
                    stepConfigLabels['RS'].append((alist[index])[0])
                    counter = counter + 1
                    stepPositions['RS'].append(counter)
                elif 'GA' in (alist[index])[0]:
                    stepYsArray['GA'].append((alist[index])[1]['replayMeanStep'])
                    stepYsErrArray['GA'].append((alist[index])[1]['replaySigmaStep'])
                    stepConfigLabels['GA'].append((alist[index])[0])
                    counter = counter + 1
                    stepPositions['GA'].append(counter)
                elif 'BO' in (alist[index])[0]:
                    stepYsArray['BO'].append((alist[index])[1]['replayMeanStep'])
                    stepYsErrArray['BO'].append((alist[index])[1]['replaySigmaStep'])
                    stepConfigLabels['BO'].append((alist[index])[0])
                    counter = counter + 1
                    stepPositions['BO'].append(counter)
            index = index + 1
        stepParams[model]=stepYsArray,stepYsErrArray,stepConfigLabels,stepPositions

        #generating series for Min Steps based comparission
        alist=bestValuesByModelStats[model]['MinStep']
        minStepConfigLabels={'RS': [], 'GA': [], 'BO': []}
        minStepYsArray={'RS': [], 'GA': [], 'BO': []}
        minStepPositions = {'RS': [], 'GA': [], 'BO': []}
        index=0
        while index<15:
            if 'RS' in (alist[index])[0]:
                minStepYsArray['RS'].append((alist[index])[1]['replayMinStep'])
                minStepConfigLabels['RS'].append((alist[index])[0])
                minStepPositions['RS'].append(index+1)
            elif 'GA' in (alist[index])[0]:
                minStepYsArray['GA'].append((alist[index])[1]['replayMinStep'])
                minStepConfigLabels['GA'].append((alist[index])[0])
                minStepPositions['GA'].append(index+1)
            elif 'BO' in (alist[index])[0]:
                minStepYsArray['BO'].append((alist[index])[1]['replayMinStep'])
                minStepConfigLabels['BO'].append((alist[index])[0])
                minStepPositions['BO'].append(index+1)
            index=index+1
        minStepParams[model]=minStepYsArray,minStepConfigLabels,minStepPositions

    gra.graphModelsComparisson(xLab,rewParams,'rewards','Mean Rewards')
    gra.graphModelsComparisson(xLab,stepParams,'steps','Mean Steps')
    gra.graphModelsComparisson(xLab,minStepParams,'minSteps','Min Steps')

def generateTimesOptStepsModelBasedGraph(modelShortName,optShortName):
    global scenariosSufix
    global optimSteps
    global results
#    optSt=[x for x in optimSteps if optShortName in x]
    params=[]
    configsLab = scenariosSufix
    referenceLabels=[]
    for opst in [x for x in optimSteps if optShortName in x]:
        #opst es por ejemplo GA20 y hay que armar las series para el graf de  IZHGA20
        #armar las series
        #graficar
        train = []
        replay = []
        referenceLabels.append(opst.replace(optShortName,""))
        datasToProcess = getResultsKeysWithCondition(modelShortName+opst)
        for dataKey in datasToProcess:
            data=results[dataKey]
            if data=={}:
                train.append(0.0)
                replay.append(0.0)
            else:
                train.append(data['trainingTime'])
                replay.append(data['replayTime'])
        params.append(replay)
        params.append(train)
    version='time'
    sugar1='replaytime'
    sugar2='trainingtime'
    gra.graphConfigurationComparisson(configsLab,referenceLabels,'Time',modelShortName+optShortName,params,sugar2,version,'training time')
    gra.graphConfigurationComparisson(configsLab,referenceLabels,'Time',modelShortName+optShortName,params,sugar1,version,'replay time')








#----------- stats generation for comparisson

def statsGenForComparisson():
    global bestValuesByModelStats
    bestValuesByModelStats = {'IandF':{'MeanRew':[],'MeanStep':[],'MinStep':[]},'FIU':{'MeanRew':[],'MeanStep':[],'MinStep':[]},'IZH':{'MeanRew':[],'MeanStep':[],'MinStep':[]}}
    global bestRewByModel
    global bestStepByModel
    global bestMinStepByModel
    bestRewByModel={'IandF':0.0,'IZH':0.0,'FIU':0.0}
    bestStepByModel={'IandF':999,'IZH':999,'FIU':999}
    bestMinStepByModel={'IandF':999,'IZH':999,'FIU':999}
    models=['IandF','IZH','FIU']
    for mod in models:
        getStatsFor(mod)


def getStatsFor(model):
    global bestValuesByModelStats
    resultsOfThatModel=[(k,v) for k, v in results.items() if (model in k and v!={})]
    #adict= sorted(resultsOfThatModel, key=lambda item: (item[1]['replayMeanRew'],item[1]['replaySigmaRew']),reverse=True)
    alist= sorted(resultsOfThatModel, key=lambda item: (item[1]['replayMeanRew']),reverse=True)
    bestValuesByModelStats[model]['MeanRew']=alist
    alist= sorted(resultsOfThatModel, key=lambda item: (item[1]['replayMeanStep']),reverse=False)
    bestValuesByModelStats[model]['MeanStep']=alist
    alist= sorted(resultsOfThatModel, key=lambda item: (item[1]['replayMinStep']),reverse=False)
    bestValuesByModelStats[model]['MinStep']=alist


#--------------- parsing results
def parseAllDatas(rootPath):
    global results
    global scenariosSufix
    global models
    global optimSteps
    global optimizers
    global pulseStats


    scenariosSufix = ['11a', '11b', '11c', '11d', '11e', '52a', '52b', '52c', '52d', '52e', '55a', '55b', '55c', '55d',
                      '55e', '105a', '105b', '105c', '105d', '105e', '1010a', '1010b', '1010c', '1010d', '1010e']
    models=['IandF','IZH','FIU']
    optimSteps=['RS2000','RS5000','RS10000','BO500','BO1000','BO1500','GA20','GA50','GA80']
    optimizers=['RS','BO','GA']
    results={}
    pulseStats={}
    parsePulseFile(rootPath)
    for mod in models:
        for opst in optimSteps:
            for scen in scenariosSufix:
                scenName=mod+opst+scen
                results[scenName]={}
    for mod in models:
        for op in optimizers:
            parseGatheredFile(rootPath,mod+op)

def parseGatheredFile(rootPath,name):
    global results
    arch = rootPath+'gatheredStats'+name
    file=open(arch)
    csvreader=csv.reader(file)
    for row in csvreader:
        rowData=parseRow(row)
        results[rowData[0]] = rowData[1]







def parseRow(row):
    global results
    toRet={}
    config=''
    if (results[row[0]+'a']=={}):
        config=row[0]+'a'
    elif (results[row[0]+'b']=={}):
        config=row[0]+'b'
    elif (results[row[0]+'c']=={}):
        config=row[0]+'c'
    elif (results[row[0] + 'd'] == {}):
        config=row[0]+'d'
    else:
        config=row[0]+'e'
    toRet['trainRew']=round(float(row[1]),3)
    toRet['replayMeanRew']=float(row[8])
    toRet['replaySigmaRew']=round(float(row[9]),3)
    toRet['replayMinStep']=float(row[5])
    toRet['replayMeanStep']=float(row[6])
    toRet['replaySigmaStep']=round(float(row[7]),3)
    toRet['trainingTime']=round(getSecondsFromStatsFileFormat(row[2])+getSecondsFromStatsFileFormat(row[3]),3)
    toRet['replayTime']=round(getSecondsFromStatsFileFormat(row[10])+getSecondsFromStatsFileFormat(row[11]),3)
    toRet['ponderedMagnitud']=0
    return config,toRet



def parsePulseFile(rootPath):
    global pulseStats
    csv.register_dialect('piper', delimiter='|')
    arch = rootPath+'pulseTimeResults'
    file=open(arch)
    csvreader=csv.reader(file,dialect='piper')
    for row in csvreader:
        pulseStats[row[0]]=[round(getSecondsFromStatsFileFormat(row[1])+getSecondsFromStatsFileFormat(row[2]),3),row[1]]
    file.close()

#------------------------------------------------------------















#------utils
def getSecondsFromStatsFileFormat(timeStr):
    mins=timeStr.split('m')[0]
    seconds=(timeStr.split('m')[1])[0:-1]
    td=timedelta(minutes=float(mins),seconds=float(seconds))
    return td.total_seconds()


def getResultsKeysWithCondition(condition):
    global results
    return [k for k, v in results.items() if condition in k]