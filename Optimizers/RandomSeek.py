import numpy as np
import datetime
import random as rng
import configparser
import os
import matplotlib.pyplot as plt




class RandomSeek:

    def __init__(self):
        self.rlengine=None
        self.distortions={}
        self.variances={}
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
        config = configparser.RawConfigParser()
        path = "../RLEngine/load.conf"
        config.read(path)
        self.batch_mode = config.get('RANDOMSEEK', 'batchmode')
        self.fit_type = config.get('RANDOMSEEK', 'fit')

    def getName(self):
        return 'RandomSeek'

    def run(self):
        log_freq=100
        self.setInitialDistortions()
        self.setInitialVariance()
        self.randonize()
        self.rlengine.model.commitNoise()
        current_return=self.rlengine.runEpisodes(self.fit_type,self.batch_mode)
        self.values[0] = current_return
        self.toPlotOrd.append(current_return)
        self.toPlotAbs.append(0)
        self.bestReward=current_return
        self.bestSolution=self.rlengine.model.clone()
        self.setBaseDistortions()
        steps_since_last_improvement = 0
        steps_since_last_improvement2 = 0
        steps = 0
        while steps < self.rlengine.steps:
            steps += 1
            self.setStepDistortions()
            self.setStepVariance()
            self.randonize()
            new_return = self.rlengine.runEpisodes(self.fit_type, self.batch_mode)
            if (new_return > current_return):
                print('Improvement! New Return: %s, old: %s' % (new_return,current_return))
                if new_return>self.bestReward :
                    #if new_return > self.bestReward and new_return > 90:
                    print('Best Improvement! New Return: %s, old: %s' % (new_return,current_return))
                    self.bestSolution=self.rlengine.model.clone()
                    self.bestReward=new_return
                    self.toPlotOrd.append(new_return)
                    self.toPlotAbs.append(steps)
                current_return = new_return
                self.rlengine.model.commitNoise()
                steps_since_last_improvement = 0
                steps_since_last_improvement2 = 0
                self.setDecresedDistortions()
            else:
                steps_since_last_improvement += 1
                steps_since_last_improvement2 +=1
                self.rlengine.model.revertNoise()
                # no improvement seen for 100 steps
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 < 300):
                    print('more than 50 steps...')
                    steps_since_last_improvement = 0
                    steps_since_last_improvement2 += 1
                    self.setIncresedDistortions()
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 > 300):
                    print('more than 300 steps without improvement...')
                    steps_since_last_improvement=0
                    steps_since_last_improvement2=0
                    self.randozieAll()
                    self.rlengine.model.commitNoise()
            if (steps % log_freq == 0):
                self.toPlotOrd.append(new_return)
                self.toPlotAbs.append(steps)
            print('paso el:'+str(steps))


    def setInitialDistortions(self):
        self.distortions['weight']=15
        self.distortions['vleak'] = 8
        self.distortions['gleak'] = 8
        self.distortions['sigma'] = 10
        self.distortions['cm'] = 10

    def setInitialVariance(self):
        self.variances['weight']=0.5
        self.variances['vleak'] = 8
        self.variances['gleak'] = 0.2
        self.variances['sigma'] = 0.2
        self.variances['cm'] = 0.1

    def setBaseDistortions(self):
        self.distortions['weight']=6
        self.distortions['vleak'] = 5
        self.distortions['gleak'] = 4
        self.distortions['sigma'] = 5
        self.distortions['cm'] = 4


    def setStepDistortions(self):
        self.distortions['weight']=rng.randint(0, self.distortions['weight'])
        self.distortions['vleak'] = rng.randint(0, self.distortions['vleak'])
        self.distortions['gleak'] = rng.randint(0, self.distortions['gleak'])
        self.distortions['sigma'] = rng.randint(0, self.distortions['sigma'])
        self.distortions['cm'] = rng.randint(0, self.distortions['cm'])

    def setStepVariance(self):
        self.variances['weight']=rng.uniform(0.01, 0.8)
        self.variances['vleak'] =  rng.uniform(0.1, 3)
        self.variances['gleak']  = rng.uniform(0.05, 0.8)
        self.variances['sigma'] = rng.uniform(0.01, 0.08)
        self.variances['cm'] = rng.uniform(0.01, 0.3)

    def setDecresedDistortions(self):
        if (self.distortions['weight'] > 6):
            self.distortions['weight'] -= 1
        if (self.distortions['sigma'] > 5):
            self.distortions['sigma'] -= 1
        if (self.distortions['vleak'] > 4):
            self.distortions['vleak'] -= 1
        if (self.distortions['gleak'] > 4):
            self.distortions['gleak'] -= 1
        if (self.distortions['cm'] > 4):
            self.distortions['cm'] -= 1

    def setIncresedDistortions(self):
        if (self.distortions['weight'] < 16):
            self.distortions['weight'] += 1
        if (self.distortions['sigma'] < 12):
            self.distortions['sigma'] += 1
        if (self.distortions['vleak'] < 8):
            self.distortions['vleak'] += 1
        if (self.distortions['gleak'] < 8):
            self.distortions['gleak'] += 1
        if (self.distortions['cm'] < 7):
            self.distortions['cm'] += 1



    def randonize(self):
        self.rlengine.model.addNoise('Weight', self.variances['weight'], self.distortions['weight'])
        self.rlengine.model.addNoise('Vleak', self.variances['vleak'], self.distortions['vleak'])
        self.rlengine.model.addNoise('Gleak', self.variances['gleak'], self.distortions['gleak'])
        self.rlengine.model.addNoise('Sigma', self.variances['sigma'], self.distortions['sigma'])
        self.rlengine.model.addNoise('Cm', self.variances['cm'], self.distortions['cm'])

    def randozieAll(self):
        self.rlengine.model.addNoise('Weight', 0.5, 26)
        self.rlengine.model.addNoise('Vleak', 10, 11)
        self.rlengine.model.addNoise('Gleak', 0.2, 11)
        self.rlengine.model.addNoise('Sigma', 0.2, 26)
        self.rlengine.model.addNoise('Cm', 0.1, 11)

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

    def saveModel(self):
        self.bestSolution.dumpModel(self.rootpath+'/model.xml')

    def plotResult(self):
        ordenadas=self.toPlotOrd
        abscisas = self.toPlotAbs
        plt.plot(abscisas, ordenadas, 'ro',label='reward')
        plt.ylabel('reward')
        plt.xlabel('steps')
        plt.legend()
        plt.title('Rewards vs Steps')
        plt.savefig(self.rootpath+'/rewards.png', bbox_inches='tight')
        plt.show()


    def recordResults(self,time):
        graphicFilesPath='file:///ariel/DataScience/Gusano/BiMPuRLE/RLEngine/'+self.rlengine.rootpath+str(self.bestReward)
        fileName = self.rlengine.rootpath + '/results.csv'
        file=open(fileName, 'a')
        line = str(self.rlengine.steps) + ',' + str(self.rlengine.batch) + ',' + str(self.rlengine.worst) + ','
        line = line +  self.fit_type+ ','+ self.batch_mode+','+str(self.rlengine.gamma) + ','
        line = line + str(time) + ',' + str(self.bestReward) +',#,%,@,'+ str(self.bestReward)
        file.write(line+','+graphicFilesPath+'/rewards.png,'+graphicFilesPath+'/hist.png'+'\n')
        file.close()


