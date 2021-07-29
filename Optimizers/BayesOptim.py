import configparser
from hyperopt import tpe
from hyperopt import anneal
from hyperopt import fmin
from hyperopt import Trials

import os
import matplotlib.pyplot as plt

import Models.IFNeuronalCircuit.ModelInterfaces as modint


class Bayes:

    def __init__(self):
        self.rootpath=None
        self.rlengine=None
        self.hyperOptSpace=None
        self.hyperOpt_algo=None
        self.hyperOpt_algo_name=''
        self.hyperOpt_trials=None
        self.hyperOpt_best_reward=0
        self.hyperOpt_best_solution=None
        self.fit_type=''
        self.batch_mode=''

    def hyperOptObjective(self, params):
        """Objective function to minimize"""
        modint.putVariablesFromHyperOpt(self.rlengine.model,params)
        fit = (self.rlengine.runEpisodes(self.fit_type,self.batch_mode))
        #print('fit:%s,f(fit):%s,g(f(fit)):%s'%(fit,f(fit),g(f(fit))))
        return f(fit)



    def initialize(self,engine):
        self.rootpath = engine.rootpath
        self.rlengine=engine
        config = configparser.RawConfigParser()
        path= "../RLEngine/load.conf"
        config.read(path)
        self.batch_mode = config.get('BAYES', 'batchmode')
        self.fit_type = config.get('BAYES', 'fit')
        self.hyperOpt_algo_name = config.get('BAYES', 'alg')
        if (self.hyperOpt_algo_name=='tpe'):
        # Create the algorithm
            self.hyperOpt_algo = tpe.suggest
        elif(self.hyperOpt_algo_name=='anneal'):
            self.hyperOpt_algo = anneal.suggest
    # Create the domain space
        self.hyperOptSpace=modint.getSpaceForHyperOpt(self.rlengine.model)
    # Create a trials object
        self.hyperOpt_trials = Trials()


    def run(self):
        best = fmin(fn=self.hyperOptObjective, space=self.hyperOptSpace,
                    algo=self.hyperOpt_algo,
                    trials=self.hyperOpt_trials,
                    max_evals=self.rlengine.steps)
        self.hyperOpt_best_reward=g(self.hyperOpt_trials.best_trial['result']['loss'])
        self.hyperOpt_best_solution=best
        print(str(self.hyperOpt_best_reward))
        print(best)


    def setPath(self):
        label=str(self.hyperOpt_best_reward)
        if not os.path.exists(self.rootpath + label):
            os.makedirs(self.rootpath + label)
        self.rootpath=self.rootpath  + label

    def saveModel(self):
        #global rlengine
        #global  hyperOpt_best_solution
        modint.putVariablesFromHyperOpt(self.rlengine.model,self.hyperOpt_best_solution)
        self.rlengine.model.dumpModel(self.rootpath+'/model.xml')

    def plotResult(self):
        ordenadas=[]
        for item in self.hyperOpt_trials.results:
            ordenadas.append(g(item['loss']))
        abscisas = [i for i in range(0,len(ordenadas))]
        plt.plot(abscisas, ordenadas, 'ro',label='reward')
        plt.ylabel('reward')
        plt.xlabel('pasos')
        plt.legend()
        plt.title('Rewards vs Pasos')
        plt.savefig(self.rootpath+'/rewards.png', bbox_inches='tight')
        plt.show()


    def saveConfig(self):
        confFile = open(self.rootpath  + '/config.prp', 'w')
        params=self.rlengine.reportParams()
        for key, value in params.items():
            confFile.write(key+'='+str(value)+'\n')
        confFile.write("algorithm=%s\n" % (self.hyperOpt_algo_name) )
        confFile.write("fitnes=%s\n" % (self.fit_type) )
        confFile.write("batch mode=%s\n" % (self.batch_mode) )
        confFile.write("steps=%s\n" % (self.rlengine.steps ) )
        confFile.close()


    def getName(self):
        return 'BayesOpt'


    def recordResults(self,time):
        graphicFilesPath='file:///ariel/DataScience/Gusano/BiMPuRLE/RLEngine/'+self.rlengine.rootpath+str(self.hyperOpt_best_reward)
        fileName = self.rlengine.rootpath + '/results.csv'
        file=open(fileName, 'a')
        line = str(self.rlengine.steps) + ',' + str(self.rlengine.batch) + ',' + str(self.rlengine.worst) + ','
        line = line +  self.fit_type+ ','+ self.batch_mode+','+str(self.rlengine.gamma) + ','
        line = line + str(time) + ',' + str(self.hyperOpt_best_reward) +',#,%,@,'+ str(self.hyperOpt_best_reward)
        file.write(line+','+graphicFilesPath+'/rewards.png,'+graphicFilesPath+'/hist.png'+'\n')
        file.close()


def f(x):
    return -x-100

def g(y):
    return -(y+100)
