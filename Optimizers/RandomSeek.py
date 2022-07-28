import configparser
import os
import matplotlib.pyplot as plt
import copy
import datetime

#this Random Seek in fact is only to IFModel
class RandomSeek:


    global modelos
    global stateToChangeTrace

    def __init__(self):
        self.rlengine=None
        self.batch_mode=''
        self.fit_type=''
        self.values={}
        self.bestSolution=None
        self.bestReward = -100
        self.rootpath=''
        self.toPlotAbs=[]
        self.toPlotOrd=[]

    def initialize(self,engine):
        self.rlengine=engine
        currDirname = os.path.dirname(__file__)
        patherDir=os.path.join(currDirname, '..')
        path=os.path.join(patherDir,"RLEngine/load.conf")
        #print(str(path))
        config = configparser.RawConfigParser()
        #path = "../RLEngine/load.conf"
        config.read(path)
        self.batch_mode = config.get('RANDOMSEEK', 'batchmode')
        self.fit_type = config.get('RANDOMSEEK', 'fit')
        global modelos
        modelos=[]

    def getName(self):
        return 'RandomSeek'




    def run(self,mode=None):
        if mode=='graphSearch':
            self.rlengine.model.updateStateToTrace()
            #print(id(self.rlengine.model))
        log_freq=100
        #modelVar=copy.deepcopy(self.rlengine.model)
        self.rlengine.modelInterface.setInitialDistortions()
        self.rlengine.modelInterface.setInitialVariance()
        self.rlengine.modelInterface.randonize(self.rlengine.model)
        if mode=='graphSearch':
            self.rlengine.model.updateStateToTrace()
            #print(id(self.rlengine.model))
        self.rlengine.model.commitNoise()
        current_return=self.rlengine.runEpisodes(self.fit_type,self.batch_mode)
        self.values[0] = current_return
        self.toPlotOrd.append(current_return)
        self.toPlotAbs.append(0)
        self.bestReward=current_return
        self.bestSolution=copy.deepcopy(self.rlengine.model)
        self.rlengine.modelInterface.setBaseDistortions()
        steps_since_last_improvement = 0
        steps_since_last_improvement2 = 0
        steps = 0
        starttime = datetime.datetime.now()
        #if mode is timed the steps parameter is intended to store the threshold time
        while steps < self.rlengine.steps:
            if mode=='timed':
                steps=(datetime.datetime.now() - starttime).total_seconds()
            else:
                steps += 1
            #self.rlengine.modelInterface.setStepDistortions()
            self.rlengine.modelInterface.setStepVariance()
            #modelVar = copy.deepcopy(self.rlengine.model)
            self.rlengine.modelInterface.randonize(self.rlengine.model)
            if mode == 'graphSearch':
                self.rlengine.model.updateStateToTrace()
                #print(id(self.rlengine.model))
            new_return = self.rlengine.runEpisodes(self.fit_type, self.batch_mode)
            #print('new ret:%s'%(new_return))
            if (new_return > current_return):
                print('Improvement! New Return: %s, old: %s' % (new_return,current_return))
                self.rlengine.model.commitNoise()
                if new_return>self.bestReward :
                    if new_return > self.bestReward :
                        print('Best Improvement! New Return: %s, old: %s' % (new_return,current_return))
                        self.bestSolution=copy.deepcopy(self.rlengine.model)
                        self.bestSolution.setName('CP'+str(steps))
                        #print(str(id(self.bestSolution)))
                        self.bestReward=new_return
                        self.toPlotOrd.append(new_return)
                        self.toPlotAbs.append(steps)
                current_return = new_return
                steps_since_last_improvement = 0
                steps_since_last_improvement2 = 0
                self.rlengine.modelInterface.setDecresedDistortions()
            else:
                steps_since_last_improvement += 1
                steps_since_last_improvement2 +=1
                self.rlengine.model.revertNoise()
                # no improvement seen for 100 steps
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 < 300):
                    print('more than 50 steps...')
                    steps_since_last_improvement = 0
                    steps_since_last_improvement2 += 1
                    self.rlengine.modelInterface.setIncresedDistortions()
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 > 300):
                    print('more than 300 steps without improvement...')
                    steps_since_last_improvement=0
                    steps_since_last_improvement2=0
                    self.rlengine.modelInterface.randozieAll(self.rlengine.model)
                    self.rlengine.model.commitNoise()
            if (steps % log_freq == 0):
                self.toPlotOrd.append(new_return)
                self.toPlotAbs.append(steps)
            print('step:'+str(steps))
        self.rlengine.model=self.bestSolution
        print("Better Soluition: "+self.rlengine.model.getName())




    def setPath(self):
        label=str(self.bestReward)
        if not os.path.exists(self.rlengine.rootpath + '/' + label):
            os.makedirs(self.rlengine.rootpath + '/' + label)
        self.rootpath=self.rlengine.rootpath + '/' + label

    def saveConfig(self):
        confFile = open(self.rootpath  + '/config.prp', 'w')
        params=self.rlengine.reportParams()
        for key, value in params.items():
            confFile.write(key+'='+str(value)+'\n')
        confFile.write("fitnes=%s\n" % (self.fit_type) )
        confFile.write("batch mode=%s\n" % (self.batch_mode) )
        confFile.write("steps=%s\n" % (self.rlengine.steps ) )
        confFile.close()

    def putOptModelIntoRLEModel(self):
        #already is in the model the optimized one
        #self.rlengine.model=self.bestSolution
        pass

    def getFolder(self):
        return  str(self.bestReward)

    #def saveModel(self):
    #    self.bestSolution.dumpModel(self.rootpath+'/model.xml')

    def plotResult(self,mode=None):
        ordenadas=self.toPlotOrd
        abscisas = self.toPlotAbs
        plt.plot(abscisas, ordenadas, 'ro',label='reward')
        plt.ylabel('reward')
        plt.xlabel('steps')
        plt.legend()
        plt.title('Rewards vs Steps')
        plt.savefig(self.rootpath+'/rewards.png', bbox_inches='tight')
        plt.show()
        if mode == 'graphSearch':
            self.rlengine.model.graphVariableTraces(self.rootpath)


    def recordResults(self,time):
        graphicFilesPath='file:///ariel/DataScience/Gusano/BiMPuRLE/RLEngine/'+self.rlengine.rootpath+str(self.bestReward)
        fileName = self.rlengine.rootpath + '/results.csv'
        file=open(fileName, 'a')
        line = str(self.rlengine.steps) + ',' + str(self.rlengine.batch) + ',' + str(self.rlengine.worst) + ','
        line = line +  self.fit_type+ ','+ self.batch_mode+','+str(self.rlengine.gamma) + ','
        line = line + str(time) + ',' + str(self.bestReward) +',#,%,@,'+ str(self.bestReward)
        file.write(line+','+graphicFilesPath+'/rewards.png,'+graphicFilesPath+'/hist.png'+'\n')
        file.close()


