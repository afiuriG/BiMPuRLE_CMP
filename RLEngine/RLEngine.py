import argparse
import os
import sys
import datetime
#from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import Utils.GraphUtils as gu

import Optimizers.RandomSeek
import Utils.GraphUtils as gra
import Utils.Shuffling as shuf
from PIL import Image
import csv
import collections


#for environments
import gym
# for Models
import Models.IFNeuronalCircuit.Model as IFmod
import Models.Izhikevich.Model as IZHmod
import Models.Haimovici.Model as HAImod
import Models.Fiuri.Model as FIUmod
import Models.IFNeuronalCircuit.ModelInterfaces as modintIF
import Models.Izhikevich.ModelInterfaces as modintIZH
import Models.Haimovici.ModelInterfaces as modintHAI
import Models.Fiuri.ModelInterfaces as modintFIU
#for optimizers
from Optimizers import RandomSeek as rs, BayesOptim as ba, GeneticAlgorithm as ga
from Models import IFNeuronalCircuit, Izhikevich
import pickle
from datetime import  timedelta

#invocation params
commandParam=''
folderParam=''
optimParam=''
modelParam=''
environParam=''
uidParam=''
batchParam=''
worstParam=''
stepsParam=''
gammaParam=''

sourceFolderIGA=''

modelUsedAsVar=None
baseModelPath=''
stepsListWhenEvaluation=[]

class RLEngine:

    def __init__(self,path,mod,env,opt,batch,worst,steps,modint,gamma=1):
        self.rootpath=path
        self.model=mod
        self.environment=env
        self.optimizer=opt
        self.batch=batch
        self.worst=worst
        self.steps=steps
        self.gamma = gamma
        self.modelInterface=modint
        #self.replayFolder=''

    def TensorRGBToImage(self, tensor):
        new_im = Image.new("RGB", (tensor.shape[1], tensor.shape[0]))
        pixels = []
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                r = tensor[y][x][0]
                g = tensor[y][x][1]
                b = tensor[y][x][2]
                pixels.append((r, g, b))
        new_im.putdata(pixels)
        return new_im

    def getModelBasePath(self):
        global baseModelPath
        return baseModelPath

    def restorePickleModel(self,fromFile):
        infile = open(fromFile, 'rb')
        newModel = pickle.load(infile)
        infile.close()
        self.model=newModel

    def dumpPickleModel(self):
        # print('Antes del preguntarle al optimizer:')
        # print(str(self.model))
        self.optimizer.putOptModelIntoRLEModel()
        # print('Despues de pregunt al optimizer:')
        # print(str(self.model))
        self.model.commitNoise()
        folder=self.optimizer.getFolder()
        filename = self.rootpath + folder+'/model'
        outfile = open(filename, 'wb')
        pickle.dump(self.model, outfile)
        outfile.close()
        print('Model saved in: ', filename)

    def train(self):
        print('se ejecuto el train')


    def optimize(self,mode=None):
        #print('Estado inicial del modelo:')
        #print(str(self.model))
        starttime = datetime.datetime.now()
        self.optimizer.initialize(self)
        self.optimizer.run(mode)
        elapsedTime = datetime.datetime.now() - starttime
        print('elapsed time: %s ' % (elapsedTime.total_seconds()))
        self.optimizer.recordResults(elapsedTime)
        self.optimizer.setPath()
        self.optimizer.saveConfig()
        self.dumpPickleModel()
        self.optimizer.plotResult(mode)
        # reward = self.run_one_episode('reward')
        # print('primera prueba run_one_episode:',reward)
        #folder = self.optimizer.getFolder()
        #self.replayFolder(folder)

    def reportParams(self):
        params={}
        params['model']=self.model.getName()
        params['environment']='MouCarCon'
        params['optimizer']=self.optimizer.getName()
        params['batch']=self.batch
        params['worst']=self.worst
        params['steps']=self.steps
        params['gamma']=self.gamma
        global commandParam
        global folderParam
        params['cmd']=commandParam
        params['folder']=folderParam
        return params



    def runEpisodes(self,fit_type,batch_mode):
        returns = np.zeros(self.batch)
        for i in range(0, self.batch):
            returns[i] = self.run_one_episode(fit_type)
        sort = np.sort(returns)
        worst_cases = sort[0:self.worst]
        # print("tw:%s" % (str(np.mean(returns))))
        allmean=np.mean(returns)
        worstmean=np.mean(worst_cases)
        toRet=None
        if(batch_mode=='all'):
            toRet=allmean
        if(batch_mode=='worst'):
            toRet=worstmean
        return toRet

    def run_one_episode(self,fit_type,mode=None):
 #       minObs=0
        maxObs=-1
        gamma = self.gamma
        total_reward = 0

        self.model.Reset()
        obs = self.environment.reset()
        observations = self.modelInterface.envObsToModelObs(obs)
        i = 1
        actionsToGraph=[]
        if mode=='debug':
            self.model.neuralnetwork.resetActivationTraces()

        #self.model.setGenerationGraphModeFolderOn(self.rootpath + '/graph')
        while 1:
            action = self.model.Update(observations,mode,False)
            actionsToGraph.append(action)
            actions = self.modelInterface.modActionToEnvAction(action)
            obs, r, done, info = self.environment.step(actions)
            #print('step:%s,action:%s,obs : %s, reward: %r' % (i, action,obs,total_reward))
            #print ('obs : %s, reward: %r'%(obs,total_reward))
            observations = self.modelInterface.envObsToModelObs(obs)
#            if obs[0]<minObs:
#                minObs=obs[0]
            if obs[0]>maxObs:
                maxObs=obs[0]
            total_reward += r * gamma
            #print('total & current reward: %s, %s'%(total_reward,r))
            if mode=='debug':
                if i==5:
                    #this is for trace graph generation only for this step
                    self.model.setGenerationGraphModeFolderOn(self.rootpath+'/graph')
            if mode=='video':
                screen = self.environment.render(mode='rgb_array')
                pic = self.TensorRGBToImage(screen)
                pic.save(self.rootpath + '/vid/img_' + str(i).zfill(5) + '.png')
            if (done):
                break
            i += 1
        #print('Return: '+str(total_reward))
        if mode=='debug':
            self.graphActions(actionsToGraph, self.rootpath + '/graph')
            self.model.neuralnetwork.writeToGraphs(self.rootpath + '/graph')
        #print('min,max,diff.rew: %s, %s, %s,%s'%(minObs,maxObs,(maxObs-minObs),total_reward))
        #print (str(i))
        if mode=='replay':
            stepsListWhenEvaluation.append(i)
        if fit_type=='reward':
            return total_reward
        elif fit_type=='position':
            return maxObs




    # to debug
    def graphActions(self,atg,folder):
        fig=plt.figure()
        times=[i for i in range(0,len(atg))]
        plt.plot(times, atg, 'ro', label='actions')
        plt.ylabel('Action value')
        plt.xlabel('Episode step')
        plt.legend()
        plt.title('Episode actions')
        fig.savefig(folder+'/actions.png', bbox_inches='tight')

    #unused i think...
    def graphSearching(self,folder):
        self.rootpath=folder
        if not os.path.exists(folder + '/graph'):
            os.makedirs(folder + '/graph')
        if not os.path.exists(folder + '/vid'):
            os.makedirs(folder + '/vid')
        totrew=self.run_one_episode('reward',True)
        print(totrew)

    def debug(self,folder):
        self.rootpath=self.rootpath+folder
        self.restorePickleModel(self.rootpath + '/model')
        if not os.path.exists(self.rootpath + '/graph'):
            os.makedirs(self.rootpath + '/graph')
        self.run_one_episode('reward',mode='debug')

    def touch(self,folder):
        self.rootpath=self.rootpath+folder
        #self.restorePickleModel(self.rootpath+'/model')
        #self.model.loadFromFile(self.rootpath+'/model.xml')
        print(self.model)
        reward=self.run_one_episode('reward')
        #gra.graphSnapshotForDiscreteModels('HA',self.model,1,True)
        print(reward)

    def modelPulseStats(self,steps):
        pulseStats=steps
        for index in range(0,pulseStats):
            self.model.runPulse()


    def graphComparisson(self,mode):
        if mode=='local':
            self.graphComparissonLocal()
        else:
            self.graphComparissonDock()

    def graphComparissonDock(self):
        scenariosSufix=['11a','11b','11c','11d','11e','52a','52b','52c','52d','52e','55a','55b','55c','55d','55e','105a','105b','105c','105d','105e','1010a','1010b','1010c','1010d','1010e']
        configsLab=[]
        rewMeans=[]
        rewSigmas=[]
        stepMeans=[]
        stepSigmas=[]
        stepMins=[]
        config=self.getConfig()
        modelAb=config["modelAbrev"]
        optimAb=config["optimizerAbrev"]
        fileNamePrefix='gatheredStats'+modelAb+optimAb
        toProcess=self.getDatasFromGatheredStatsFile(self.rootpath,fileNamePrefix)



        gra.graphComparissonRew(configsLab,rewMeans,rewSigmas,self.rootpath)
        gra.graphComparissonSteps(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)
        #gra.graphComparissonTrainTime(configsLab,rewMeans,rewSigmas,self.rootpath)
        #gra.graphComparissonTestTime(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)
        #gra.graphComparissonALL(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)



    def graphComparissonDockOld(self):
        scenariosSufix=['11a','11b','11c','11d','11e','52a','52b','52c','52d','52e','55a','55b','55c','55d','55e','105a','105b','105c','105d','105e','1010a','1010b','1010c','1010d','1010e']
        files = os.listdir(self.rootpath)
        configsLab=[]
        stepMeans=[]
        stepSigmas=[]
        stepMins=[]
        rewTrain=[]
        rewMeans=[]
        rewSigmas=[]
        config=self.getConfig()
        modelAb=config["modelAbrev"]
        optimAb=config["optimizerAbrev"]
        if modelAb=='IandF' :
             modelLabel='IF'
        else:
             modelLabel=modelAb
        if optimAb=='BO' :
             optimconfigs=['500','1000','1500','2000']
        elif optimAb=='RS' :
             optimconfigs = ['2000','5000','10000','25000']
        elif optimAb=='GA' :
             optimconfigs = ['20', '50', '80', '110']
        else:
             optimconfigs = []
        fileNamePrefix='timeResults'+modelLabel+optimAb
        timeResultsFileNames=[]
        for oc in optimconfigs:
            timeResultsFileNames.append(fileNamePrefix+oc)
        toProcess=self.getDatasFromTimeResultsFile(self.rootpath,timeResultsFileNames)
        #toProcessOrdered=collections.OrderedDict(sorted(toProcess.items()))
        for opConf in optimconfigs:
            for i in range(0,25):
                key=modelAb+optimAb+opConf+scenariosSufix[i]
                folder=str(toProcess[key]).partition('/')[2]
                if folder=='0.0':
                    rewTrain.append(0.0)
                    rewMeans.append(0.0)
                    rewSigmas.append(0.0)
                    stepMeans.append(0.0)
                    stepSigmas.append(0.0)
                    stepMins.append(0.0)
                else:
                    datas = self.getDatasFromReplayFile(self.rootpath + folder + '/replay.txt')
                    rewTrain.append(float(str(toProcess[key]).partition('/')[2]))
                    rewMeans.append(float(datas['rew_mean'].partition('[')[2]))
                    rewSigmas.append(float(datas['rew_sigma']))
                    stepMeans.append(float(datas['steps_mean']))
                    stepSigmas.append(float(datas['steps_sigma']))
                    stepMins.append(float(datas['steps_min']))
                configsLab.append(opConf+scenariosSufix[i])
                #print(key+':'+str(toProcess[key]).partition('/')[2])
        gra.graphComparissonRew(configsLab,rewMeans,rewSigmas,self.rootpath)
        gra.graphComparissonSteps(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)
        #gra.graphComparissonTrainTime(configsLab,rewMeans,rewSigmas,self.rootpath)
        #gra.graphComparissonTestTime(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)
        #gra.graphComparissonALL(configsLab,stepMeans,stepSigmas,stepMins,self.rootpath)






    #need check to know if is still working
    def graphComparissonLocal(self):
        self.rootpath = self.rootpath + '../../../../Presentation/Reportes/'
        files=os.listdir(self.rootpath)
        configs=[]
        rewMeans=[]
        rewSigmas=[]
        stepMeans=[]
        stepSigmas=[]
        mins=[]
        lab=[]
        for f in files:
            if f.find('NOPROC')==-1:
               underscoreidx=f.find('_')
               toretu=f[underscoreidx+1:]
               configs.append(toretu)
               print('Process: '+toretu)
               datas=self.getDatasFromReplayFile(self.rootpath+f+'/replay.txt')
               rewMeans.append(float(datas['rew_mean']))
               rewSigmas.append(float(datas['rew_sigma']))
               stepMeans.append(float(datas['steps_mean']))
               stepSigmas.append(float(datas['steps_sigma']))
               mins.append(float(datas['steps_min']))
               #lab.append(f)
            else:
               print('Will not be processed: ' + f)
        gra.graphComparissonRew(configs,rewMeans,rewSigmas,self.rootpath)
        gra.graphComparissonSteps(configs,stepMeans,stepSigmas,mins,self.rootpath)
        #gra.graphComparissonStepsMin()

    def getDatasFromReplayFile(self,file):
        try:
            with open(file) as csv_file:
                toRet=None
                csv_reader = csv.reader(csv_file, delimiter=',',)
                for row in csv_reader:
                    toRet={'rew_mean':row[7],'rew_sigma':row[9],'steps_mean':row[10],'steps_sigma':row[12],'steps_min':row[13]}
                csv_file.close()
        except:
            toRet = {'rew_mean': 'x[0.0', 'rew_sigma': '0.0', 'steps_mean': '0.0', 'steps_sigma': '0.0',
                     'steps_min': '0.0'}
        return toRet

    def getDatasFromGatheredStatsFile(self,rootpath,f):
        toRet = {}
        toRetList=[]
        gatheredFile=rootpath+'../STATS/'+f
        with open(gatheredFile) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',',)
                for row in csv_reader:
                    scenario=row[0]
                    reward=round(float(row[1]),3)
                    userSeconds=self.getSecondsFromStatsFileFormat(row[2])
                    systemSeconds=self.getSecondsFromStatsFileFormat(row[3])
                    systemTrainTime=round(userSeconds+systemSeconds,3)
                    wallTrainTime=self.getSecondsFromStatsFileFormat(row[4])
                    minSteps=float(row[5])
                    mediaSteps=float(row[6])
                    sigmaSteps=float(row[7])
                    meanRew=row[8]
                    sigmaRew=row[9]
                    userSeconds=self.getSecondsFromStatsFileFormat(row[10])
                    systemSeconds=self.getSecondsFromStatsFileFormat(row[11])
                    systemReplayTime=round(userSeconds+systemSeconds,3)
                    wallReplayTime=self.getSecondsFromStatsFileFormat(row[12])
                    toRetList.append({'scenario':scenario,'reward':reward,
                                      'systemTrainTime':systemTrainTime,'wallTrainTime':wallTrainTime,
                                      'minSteps':minSteps,'mediaSteps':mediaSteps,'sigmaSteps':sigmaSteps,
                                      'meanRew':meanRew,'sigmaRew':sigmaRew,'systemReplayTime':systemReplayTime,
                                      'wallReplayTime':wallReplayTime})
                    # if not('-' in row[4] or '0.0' in row[4]):
                    #     toRet[row[0]]=row[4]
                    # else:
                    #     toRet[row[0]] = 'IandF_MouCarCon_BO/0.0'
                csv_file.close()
        return toRet

    def getSecondsFromStatsFileFormat(self,timeStr):
        mins=timeStr.split('m')[0]
        seconds=(timeStr.split('m')[1])[0:-1]
        td=timedelta(minutes=float(mins),seconds=float(seconds))
        return td.total_seconds()



    def recover(self,folder):
        self.rootpath=self.rootpath+folder
        self.model.loadFromFile(self.rootpath+'/model.xml')
        path = self.rootpath+'/model'
        outfile = open(path, 'wb')
        pickle.dump(self.model, outfile)
        outfile.close()
        print('ready')

    def printModel(self,folder):
        self.rootpath=self.rootpath+folder
        self.restorePickleModel(self.rootpath + '/model')
        print(self.model)
        print('ready')

    def rmResultsFolder(self,folder):
        if folder=='DIR':
            os.system('rm -rf ' + self.rootpath)
            print('deleted: '+ self.rootpath)
        else:
            self.rootpath=self.rootpath+folder
            os.system('rm -rf '+self.rootpath)
            print('deleted: '+self.rootpath)



    def video(self,folder):
        self.rootpath=folder
        self.restorePickleModel(self.rootpath + '/model')
        if not os.path.exists(folder + '/vid'):
            os.makedirs(folder + '/vid')
        self.run_one_episode('reward',mode='video')
        path = self.rootpath + '/vid/'
        command = "ffmpeg -framerate 24 -i " + path + "img_%05d.png " + path + "output.mp4"
        result = os.system(command)
        if result != 0:
            print('Error al generar video')
        else:
            print('Video generated correctly')
            command = "rm -rf " + path + "*.png"
            result = os.system(command)
            if result == 0:
                print('Correctamente borrado de .png files')
            else:
                print('Error en borrado de .png files con code: %s' % (result))

    def replay(self,folder):
        noTouchList=[]
        toTouchList=[]
        if folder=='all':
            print('corrio todos los que no esten corridos')
            with open(self.rootpath + 'results.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[8]=='#':
                        toTouchList.append(row)
                    else:
                        noTouchList.append(row)
            csv_file.close()
            for row in toTouchList:
                stats = self.replayFolder(row[11])
                row[8]=stats[0]
                row[9]=stats[1]
                row[10]=stats[2]
        else:
            print('Only one folder to run')
            with open(self.rootpath + 'results.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[11]==folder:
                        toTouchList.append(row)
                    else:
                        noTouchList.append(row)
            csv_file.close()
            stats=self.replayFolder(folder)
            toTouchList[0][8] = stats[0]
            toTouchList[0][9] = stats[1]
            toTouchList[0][10] = stats[2]

        with open(self.rootpath + 'results.csv', mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for row in noTouchList:
                writer.writerow(row)
            for row in toTouchList:
                writer.writerow(row)
        csv_file.close()
        print('Replay Finished')

    def replayFolder(self,folder):
        currentFolder=self.rootpath+folder
        replaylog = open(currentFolder + '/replay.txt', 'w')
        #is it util?
        if folder=='NullHypotesis':
            self.modelInterface.randonizeModel(self.model)
        else:
            self.restorePickleModel(currentFolder + '/model')
        #print('Starts the replay with model:')
        #print(str(self.model))
        eval = self.evaluate_avg(currentFolder)
        replaylog.write('Mean Reward, varianza, desv standar, mean Steps,  steps varianza,  min steps, steps desv standar, steps median : ' + str(eval) + '\n')
        print('Folder: '+currentFolder)
        print('Average Reward: ' + str(eval))
        replaylog.close()
        return eval

    def getConfig(self):
        if isinstance(self.model,HAImod.Model):
            model='Haimovici'
            modelAb='HAI'
        if isinstance(self.model,Izhikevich.Model.Model):
            model='Izhikevich'
            modelAb = 'IZH'
        if isinstance(self.model,IFNeuronalCircuit.Model.Model):
            model='Integrate & Fire'
            modelAb='IandF'
        if isinstance(self.model,FIUmod.Model):
            model='Fiuri'
            modelAb='FIU'

        if isinstance(self.optimizer,Optimizers.RandomSeek.RandomSeek):
            optim='Random Seek'
            optimAb='RS'
        if isinstance(self.optimizer,Optimizers.GeneticAlgorithm.GeneticAlgorithm):
            optim='Genetic Algorithm'
            optimAb = 'GA'
        if isinstance(self.optimizer,Optimizers.BayesOptim.Bayes):
            optim='Bayes Optimization'
            optimAb='BO'

        return {"modelLab":model,"optimizerLab":optim,"modelAbrev":modelAb,"optimizerAbrev":optimAb}




    def evaluate_avg(self,folder):
        N = 1000
        #model=''
        # if isinstance(self.model,HAImod.Model):
        #     model='Haimovici'
        # if isinstance(self.model,Izhikevich.Model.Model):
        #     model='Izhikevich'
        # if isinstance(self.model,IFNeuronalCircuit.Model.Model):
        #     model='Integrate & Fire'
        # if isinstance(self.optimizer,Optimizers.RandomSeek.RandomSeek):
        #     optim='Random Seek'
        # if isinstance(self.optimizer,Optimizers.GeneticAlgorithm.GeneticAlgorithm):
        #     optim='Genetic Algorithm'
        # if isinstance(self.optimizer,Optimizers.BayesOptim.Bayes):
        #     optim='Bayes Optimization'
        config=self.getConfig()
        model=config["modelLab"]
        optim=config["optimizerLab"]
        returns = np.zeros(N)
        for i in range(0, N):
            returns[i] = self.run_one_episode('reward',mode='replay')
        gra.graphReplayRewardsHist(optim,model,returns, folder)
        mean = np.mean(returns)
        var = np.var(returns, ddof=0)
        std = np.std(returns, ddof=0)
        gra.graphReplayStepsHist(optim,model,stepsListWhenEvaluation, folder)
        minStepsVal=min(stepsListWhenEvaluation)
        #print('[min,max]steps:%s,%s'%(minStepsVal,maxStepsVal))
        stepsMean = np.mean(stepsListWhenEvaluation)
        stepsVar = np.var(stepsListWhenEvaluation, ddof=0)
        stepsStd = np.std(stepsListWhenEvaluation, ddof=0)
        stepsMedian = np.median(stepsListWhenEvaluation)
        return [mean, var, std,stepsMean, stepsVar, stepsStd,minStepsVal,stepsMedian]








def createEngine(path,mod,env,opt,batch,worst,steps,gamma):
    if (env=='MouCarCon'):
        environment= gym.make("MountainCarContinuous-v0")
    else:
        print ('The environment is wrong.')
    if (mod=='IandF'):
        model=createIandFModel()
        modint=modintIF
    elif (mod=='IZH'):
        model=createIzhikevichModel()
        modint=modintIZH
    elif (mod == 'HAI'):
        model = createHaimoviciModel()
        modint = modintHAI
    elif (mod == 'FIU'):
        model = createFiuriModel()
        modint = modintFIU
    elif (mod == 'GandH'):
        model = createHaimoviciModel()
        modint = modintHAI
        model.setName('GreemberAndHasting')
    elif (mod == 'QN'):
        model=''
        model = createHaimoviciModel()
        modint = modintHAI
        #model = createQNModel()
        #modint = modintQN
        #model.setName('QLearning')
        print('se creo el modelo QN')
    else:
        print('The model name is wrong.')
    if(opt=='RS'):
        optimizer=rs.RandomSeek()
    elif (opt=='GA'):
        optimizer=ga.GeneticAlgorithm()
    elif (opt=='BO'):
        optimizer=ba.Bayes()
    elif (opt == 'IGA'):
        global sourceFolderIGA
        optimizer = ga.GeneticAlgorithm(source=sourceFolderIGA)
    else:
        print ('Wrong optimizer.')
    engine=RLEngine(path,model,environment,optimizer,batch,worst,steps,modint,gamma)
    return engine


def createIandFModel():
    global baseModelPath
    currDirname = os.path.dirname(__file__)
    path = os.path.join(currDirname, "BaseModels/TWLetchBase.xml")
    baseModelPath=path
    model = IFmod.Model('IandF')
    model.loadFromFile(baseModelPath)
    global modelUsedAsVar
    modelUsedAsVar = model
    model = IFmod.Model('IandF')
    model.loadFromFile(baseModelPath)
    return model

def createIzhikevichModel():
    dirname = os.path.dirname(__file__)
    global baseModelPath
    baseModelPath = os.path.join(dirname, 'BaseModels/TWFiuriBaseIZH.xml')
    model = IZHmod.Model('IZH')
    model.loadFromFile(baseModelPath)
    global modelUsedAsVar
    modelUsedAsVar = model
    model = IZHmod.Model('IZH')
    model.loadFromFile(baseModelPath)
    return model


def createHaimoviciModel():
    dirname = os.path.dirname(__file__)
    global baseModelPath
    baseModelPath = os.path.join(dirname, 'BaseModels/TWFiuriBaseHAI.xml')
    model = HAImod.Model('HAI')
    model.loadFromFile(baseModelPath)
    global modelUsedAsVar
    modelUsedAsVar = model
    model = HAImod.Model('HAI')
    model.loadFromFile(baseModelPath)
    return model

def createFiuriModel():
    dirname = os.path.dirname(__file__)
    global baseModelPath
    baseModelPath = os.path.join(dirname, 'BaseModels/TWFiuriBaseFIU.xml')
    model = FIUmod.Model('FIU')
    model.loadFromFile(baseModelPath)
    global modelUsedAsVar
    modelUsedAsVar = model
    model = FIUmod.Model('FIU')
    model.loadFromFile(baseModelPath)
    return model




def run():
    global rootpath
    rootpath = ''
    missing=[]
    rootpath = rootpath + 'uid.' + uidParam + '/'
    if (modelParam == ''):
        missing.append('mod')
    else:
        rootpath = rootpath + modelParam + '_'

    if (environParam == ''):
        missing.append('env')
    else:
        rootpath = rootpath + environParam + '_'
    if (optimParam == ''):
        missing.append('opt')
    else:
        rootpath = rootpath + optimParam + '/'
    if(len(missing)==0):
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
            os.makedirs(rootpath+'NullHypotesis')
            createResultsCsv(rootpath)
            createResultsCsv(rootpath+'NullHypotesis')
        gotCommand=False
        if (commandParam == 'optimize'):
            print('optimizar con rootpath: '+rootpath)
            #will be used only if opt=IGA
            global sourceFolderIGA
            sourceFolderIGA=folderParam
            engine=createEngine(rootpath,modelParam,environParam,optimParam,batchParam,worstParam,stepsParam,gammaParam)
            engine.optimize()
            gotCommand=True
        if (commandParam == 'train'):
            print('train con rootpath: '+rootpath)
            engine=createEngine(rootpath,modelParam,environParam,optimParam,batchParam,worstParam,stepsParam,gammaParam)
            engine.train()
            gotCommand=True
        if (commandParam == 'optimizeNull'):
            rootpath=rootpath+'NullHypotesis/'
            print('optimize con null hypotesis: ' + rootpath )
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            #gu.graphSnapshot('IZ',engine.model,0,True)
            shuf.shuffle(engine.model)
            engine.optimize()
            engine.model.generationGraphModeFolder=rootpath+engine.optimizer.getFolder()
            if isinstance(engine.model,IZHmod.Model):
                gu.graphSnapshot('IZ',engine.model,0,True)
            if isinstance(engine.model,IFmod.Model):
                gu.graphSnapshot('IF',engine.model,0,True)
            if isinstance(engine.model,HAImod.Model):
                gu.graphSnapshot('HA',engine.model,0,True)
            if isinstance(engine.model,FIUmod.Model):
                gu.graphSnapshot('FI',engine.model,0,True)
            gotCommand = True
        if (commandParam == 'timedoptimize'):
            print('timeoptimizar con rootpath: ' + rootpath)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam,
            stepsParam, gammaParam)
            engine.optimize(mode='timed')
            gotCommand = True
        if (commandParam == 'debug'):
            print('debug con rootpath: ' + rootpath)
            engine=createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.debug(folderParam)
            gotCommand = True
        if (commandParam == 'graphS'):
            #This is only to RS!
            print('graphS con rootpath: ' + rootpath)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.optimize(mode='graphSearch')
            gotCommand = True
        if (commandParam == 'replay'):
            print('replay con rootpath: ' +rootpath+folderParam)
            engine=createEngine(rootpath,modelParam, environParam, optimParam, batchParam,worstParam, stepsParam,gammaParam)
            engine.replay(folderParam)
            gotCommand = True
            #don't know if is still used...
        if (commandParam == 'replayNull'):
            print('replay con null hypotesis on params values: ' + rootpath + 'NullHypotesis')
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.replayFolder('NullHypotesis')
            gotCommand = True
        if (commandParam == 'touch'):
            print('touch con rootpath: ' + rootpath + folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.touch(folderParam)
            gotCommand = True
        if (commandParam == 'modelPulseStats'):
            print('modelPulseStats con rootpath: ' + rootpath)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.modelPulseStats(stepsParam)
            gotCommand = True
        if (commandParam == 'recoverOldModel'):
            print('recover model from xml format: ' + rootpath + folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,
                                  gammaParam)
            engine.recover(folderParam)
            gotCommand = True
        if (commandParam == 'printModel'):
            print('print model from pickle format: ' + rootpath + folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,
                                  gammaParam)
            engine.printModel(folderParam)
            gotCommand = True
        if (commandParam == 'graphComparisson'):
            print('graph configurations comparison: ' + rootpath + folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,
                                  gammaParam)
            engine.graphComparisson('docker')
            gotCommand = True
        if (commandParam == 'rmFolder'):
            print('remove directory: ' + rootpath + folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,
                                  gammaParam)
            engine.rmResultsFolder(folderParam)
            gotCommand = True
        if (commandParam == 'video'):
            print('generate video con rootpath: ' + rootpath+folderParam)
            engine = createEngine(rootpath, modelParam, environParam, optimParam, batchParam, worstParam, stepsParam,gammaParam)
            engine.video(rootpath+folderParam)
            gotCommand = True
        if not gotCommand:
            print('Not supported command: %s'%(commandParam))
    else:
        print('Missing parameter/s: %s' %(missing))
    print('rootpath: '+ rootpath)

def createResultsCsv(path):
    fileName=path + '/results.csv'
    file=open(fileName, 'w')
    file.write('pasos,batch,worst,fitfunc,batchmode,gamma,elapsed,reward,avgerage,varianza,stddesv,folder,graphic,histogram\n')
    file.close()

def main(params):
    global commandParam,folderParam,optimParam,modelParam,environParam,uidParam,batchParam,worstParam,stepsParam,gammaParam
    commandParam = params['cmd']
    folderParam = params['folder']
    optimParam = params['opt']
    modelParam = params['mod']
    environParam = params['env']
    uidParam = params['uid']
    batchParam = int(params['batch'])
    worstParam = int(params['worst'])
    stepsParam = int(params['steps'])
    gammaParam = float(params['gamma'])
    run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default="all")
    parser.add_argument('--cmd', default='', type=str)
    parser.add_argument('--opt', default='', type=str)
    parser.add_argument('--mod', default='', type=str)
    parser.add_argument('--env', default='', type=str)
    parser.add_argument('--uid', default="0")
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--worst', default=1, type=int)
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument('--gamma', default=1, type=float)
    args = parser.parse_args()
    commandParam = args.cmd
    folderParam = args.folder
    optimParam = args.opt
    modelParam = args.mod
    environParam = args.env
    uidParam = args.uid
    batchParam = args.batch
    worstParam = args.worst
    stepsParam = args.steps
    gammaParam = args.gamma

    run()
    #--mod IandF --env MouCarCon --opt BO --cmd optimize --folder all --steps 1000 --batch 5 --worst 5 --gamma 1.0
    #--mod IandF --env MouCarCon --opt RS --cmd optimize --folder all --steps 10000 --batch 5 --worst 5 --gamma 1.0
    #--mod IandF --env MouCarCon --opt IGA --cmd optimize --folder uid.0/IGASeqSource/ --steps 70 --batch 5 --worst 5 --gamma 1.0
    #--mod IandF --env MouCarCon --opt GA --cmd replay --folder all --steps 70 --batch 5 --worst 5 --gamma 1.0

    #--mod IZH --env MouCarCon --opt BO --cmd debug --folder all --steps 1000 --batch 1 --worst 1 --gamma 1.0
    #only generate the trace graph for stpe 500 on only one episode
    #--mod IZH --env MouCarCon --opt GA --cmd graph --folder -23.274241826982124 --steps 10 --batch 1 --worst 1 --gamma 1.0

    #--mod IZH --env MouCarCon --opt BO --cmd touch --folder 95.53057470849589 --steps 1000 --batc 5 --worst 5 --gamma 1.0
    #--mod IZH --env MouCarCon --opt RS --cmd replay --folder NullHypotesis/96.68020310445371  --steps 5 --batch 1 --worst 1 --gamma 1.0