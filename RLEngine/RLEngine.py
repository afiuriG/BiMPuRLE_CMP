import argparse
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
#import importlib

#for environments
import gym
# for Models
import Models.IFNeuronalCircuit.Model as IFmod
import Models.IFNeuronalCircuit.ModelInterfaces as modint
#for optimizers
from Optimizers import RandomSeek as rs, BayesOptim as ba, GeneticAlgorithm as ga

modelUsedAsVar=None

class RLEngine:

    def __init__(self,path,mod,env,opt,batch,worst,steps,gamma=1):
        self.rootpath=path
        self.model=mod
        self.environment=env
        self.optimizer=opt
        self.batch=batch
        self.worst=worst
        self.steps=steps
        self.gamma = gamma
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

    def optimize(self):
        starttime = datetime.datetime.now()
        self.optimizer.initialize(self)
        self.optimizer.run()
        elapsedTime = datetime.datetime.now() - starttime
        print('elapsed time: %s' % (elapsedTime))
        self.optimizer.recordResults(elapsedTime)
        self.optimizer.setPath()
        self.optimizer.saveConfig()
        self.optimizer.saveModel()
        self.optimizer.plotResult()

    def reportParams(self):
        params={}
        params['model']=self.model.getName()
        params['environment']='MouCarCon'
        params['optimizer']=self.optimizer.getName()
        params['batch']=self.batch
        params['worst']=self.worst
        params['steps']=self.steps
        params['gamma']=self.gamma
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

    def run_one_episode(self,fit_type,mode=False):
        minObs=0
        maxObs=-1
        gamma = self.gamma
        total_reward = 0

        self.model.Reset()
        obs = self.environment.reset()
        observations = modint.envObsToModelObs(obs)
        i = 0
        while 1:
            action = self.model.Update(observations,0.1,10,False)
            actions = modint.modActionToEnvAction(action)
            obs, r, done, info = self.environment.step(actions)
            #print ('obs : %s, reward: %r'%(obs,r))
            observations = modint.envObsToModelObs(obs)
            if obs[0]<minObs:
                minObs=obs[0]
            if obs[0]>maxObs:
                maxObs=obs[0]
            total_reward += r * gamma
            #print('total y current reward: %s, %s'%(total_reward,r))
            if mode:
                screen = self.environment.render(mode='rgb_array')
                pic = self.TensorRGBToImage(screen)
                pic.save(self.rootpath + '/vid/img_' + str(i).zfill(5) + '.png')
            if (done):
                break
            i += 1
        # print('Return: '+str(total_reward))
        #print('min,max,diff.rew: %s, %s, %s,%s'%(minObs,maxObs,(maxObs-minObs),total_reward))
        #print (str(i))
        if fit_type=='reward':
            return total_reward
        elif fit_type=='position':
            return maxObs

    def video(self,folder):
        self.rootpath=folder
        if not os.path.exists(folder + '/vid'):
            os.makedirs(folder + '/vid')
        #self.run_one_episode('reward',True)
        #path = self.rootpath + '/vid/'
        #El ampersand rompe el path hay que buscar la forma de escaparlo!
        path='/ariel/DataScience/Gusano/BiMPuRLE/RLEngine/uid.0/I\&F_MouCarCon_RS/-18.919477234867337/vid/'
        command = "ffmpeg -framerate 24 -i " + path + "img_%05d.png " + path + "output.mp4"
        #command="ls"
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
            print('corrio solo un folder')
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
        self.model.loadFromFile(currentFolder+'/model.xml')
        eval = self.evaluate_avg(currentFolder)
        replaylog.write('Mean Reward, varianza, desv standar: ' + str(eval) + '\n')
        print('Average Reward: ' + str(eval))
        replaylog.close()
        return eval



    def evaluate_avg(self,folder):
        N = 1000
        returns = np.zeros(N)
        for i in range(0, N):
            returns[i] = self.run_one_episode('reward',False)
        mean = np.mean(returns)
        var = np.var(returns, ddof=0)
        std = np.std(returns, ddof=0)
        print('folder %s: (%s,%s)'%(folder,mean,std))
        plt.hist(returns)
        plt.savefig(folder + '/hist.png')
        plt.show()
        return [mean, var, std]


def createEngine(path,mod,env,opt,batch,worst,steps,gamma):
    if (env=='MouCarCon'):
        environment= gym.make("MountainCarContinuous-v0")
    else:
        print ('The environment is wrong.')
    if (mod=='I&F'):
        model = IFmod.Model('I&F')
        model.loadFromFile('Letchner/TWLetchBase.xml')
        global modelUsedAsVar
        modelUsedAsVar=model
        model = IFmod.Model('I&F')
        model.loadFromFile('Letchner/TWLetchBase.xml')
    else:
        print('The model name is wrong.')
    if(opt=='RS'):
        optimizer=rs.RandomSeek()
    elif (opt=='GA'):
        optimizer=ga.GeneticAlgorithm()
    elif (opt=='BO'):
        optimizer=ba.Bayes()
    else:
        print ('Wrong optimizer.')
    engine=RLEngine(path,model,environment,optimizer,batch,worst,steps,gamma)
    return engine








def run():
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
    global rootpath
    rootpath = ''
    missing=[]
    rootpath = rootpath + 'uid.' + args.uid + '/'
    if (args.mod == ''):
        missing.append('mod')
    else:
        rootpath = rootpath + args.mod + '_'

    if (args.env == ''):
        missing.append('env')
    else:
        rootpath = rootpath + args.env + '_'
    if (args.opt == ''):
        missing.append('opt')
    else:
        rootpath = rootpath + args.opt + '/'
    if(len(missing)==0):
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
            createResultsCsv(rootpath)
        if (args.cmd == 'optimize'):
            print('optimizar con rootpath: '+rootpath)
            engine=createEngine(rootpath,args.mod,args.env,args.opt,args.batch,args.worst,args.steps,args.gamma)
            engine.optimize()
        elif (args.cmd == 'replay'):
            print('replay con rootpath: ' +rootpath+args.folder)
            engine=createEngine(rootpath,args.mod,args.env,args.opt,args.batch,args.worst,args.steps,args.gamma)
            engine.replay(args.folder)
        elif (args.cmd == 'video'):
            print('generate video con rootpath: ' + rootpath+args.folder)
            engine = createEngine(rootpath, args.mod, args.env, args.opt, args.batch, args.worst, args.steps,
                                  args.gamma)
            engine.video(rootpath+args.folder)
        else:
            print('Not supported command: %s'%(args.cmd))
    else:
        print('Missing parameter/s: %s' %(missing))
    print('rootpath: '+ rootpath)

def createResultsCsv(path):
    fileName=path + '/results.csv'
    file=open(fileName, 'w')
    file.write('pasos,batch,worst,fitfunc,batchmode,gamma,elapsed,reward,avgerage,varianza,stddesv,folder,graphic,histogram\n')
    file.close()



if __name__ == "__main__":
    run()