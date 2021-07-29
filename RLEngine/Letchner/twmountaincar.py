#from OpenGL import GLU
import gym
import numpy as np
#import pybnn
import random as rng
from PIL import Image
import datetime
import os
import argparse
import Models.IFNeuronalCircuit.Model as mod
import matplotlib.pyplot as plt
import csv
import sys
from Optimizers import BayesOptim as ba, GeneticAlgorithm as ga


#####This file has errors and is old, when was working was the version Python for  Letchner paper.
class TWsearchEnv:

    def __init__(self, env, filter_len, mean_len,steps):
        self.env = env
        self.filter_len = filter_len
        self.mean_len = mean_len
        #self.twModel = mod.Model('IF_Letch300')
        #self.twModel = mod.Model('GenAlgOptim')
        #self.twModel = mod.Model('Pruebas')
        self.twModel = mod.Model('Bayes')

        self.optimSteps=steps
        self.rootPath = self.twModel.name+'/results/fil_' + str(self.filter_len) + '_mean_' + str(self.mean_len) + '_steps_' + str(
            self.optimSteps)

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

    def set_observations_for_lif(self, obs, observations):
        observations[0] = float(obs[0])
        observations[1] = float(obs[1])

    def run_one_episode(self,do_render=False,model=None,mode='RS'):
        total_reward = 0
        minObs=0
        maxObs=-1
        currModel=None
        if model==None:
            currModel=self.twModel
        else:
            currModel = model

        obs = self.env.reset()
        currModel.Reset()
        logUpd=False
        #if (do_render):
            #modelLog = open('log/oneEpisode.log', 'w')
            #modelLog.write("Arranca run_one_episode" + '\n')
            #logUpd=True
            #rewardlog = open('log/rewardlog.log', 'w')
            #rewardlog.write("Arranca run_one_episode" + '\n')
            #self.lif.DumpClear('lif-dump.csv')

        observations = []
        action = 0
        for i in range(0, 2):
            observations.append(float(0))

        self.set_observations_for_lif(obs, observations)
        actions = np.zeros(1)
        #para mi este esta de mas
        #self.lif.Update(observations, 0.01, 10)

        total_reward = 0
        #total_reward = np.zeros(1)
        gamma = 1.0
        time = 0.0

        start_pos = 0
        has_started = False
        i = 0
        #mepa que esta de mas
        #done2 = False
        while 1:
            #action = self.lif.Update(observations, 0.01, 10)
            action = currModel.Update(observations,0.1,10,logUpd)
            #el autor usaba asi action action[0], a mi me da action como un float directamente
            actions[0] = action
            obs, r, done, info = self.env.step(actions)
            #print ('obs : %s, reward: %r'%(obs,r))
            self.set_observations_for_lif(obs, observations)
            if obs[0]<minObs:
                minObs=obs[0]
            if obs[0]>maxObs:
                maxObs=obs[0]
            total_reward += r * gamma
            time += 0.0165
            #print('total y current reward: %s, %s'%(total_reward,r))
            #viene false....
            if (do_render):
                #modelLog.write("time: %s, accion: %s, -> reward: %s, obs: %s, totR: %s " % (time,action, r, observations, total_reward) + '\n')
                #modelLog.flush()
            #    rewardlog.write(str(total_reward) + '\n')
            #    rewardlog.flush()
            #    self.lif.DumpState('lif-dump.csv')
            #    self.env.render()
                #print("R: {:0.3f}".format(float(total_reward)))
                print("R: %s,, X: %s, v: %s"%(total_reward,obs[0],obs[1]))

                screen = self.env.render(mode='rgb_array')
                print('Img shape: '+str(screen.shape))
                pic = self.TensorRGBToImage(screen)
                pic.save(self.rootPath+'/vid/img_'+str(i).zfill(5)+'.png')
                phi = np.arcsin(obs[0])
                #print('Obs: '+str(phi)+', '+str(obs[1])+' Act: '+str(actions[0]))
                #ffmpeg -framerate 24 -i img_%05d.png output.mp4

                if (time >= 16.5):
                    return total_reward
            if (done):
                break
            i += 1
        # print('Return: '+str(total_reward))
        #if (do_render):
            #modelLog.close()
        #print('min,max,diff.rew: %s, %s, %s,%s'%(minObs,maxObs,(maxObs-minObs),total_reward))
        #print (str(i))
        if mode=='RS':
            return total_reward
        else:
            return maxObs

    def run_multiple_episodesGA(self,indiv):
        returns = np.zeros(self.filter_len)
        for i in range(0, self.filter_len):
            returns[i] = self.run_one_episode(False,indiv,'RS')
        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]
        #print("tw:%s" % (str(np.mean(returns))))
        return [np.mean(worst_cases), np.mean(returns)]

    def evaluate_avg(self):

        N = 1000
        returns = np.zeros(N)
        for i in range(0, N):
            returns[i] = self.run_one_episode(False,'reward')
        mean = np.mean(returns)
        var = np.var(returns,ddof=0)
        std = np.std(returns,ddof=0)
        plt.hist(returns)
        plt.savefig(self.rootPath + '/hist.png')
        plt.show()
        return [mean,var,std]

    def run_multiple_episodes(self,do_log=False,indiv=None):
        returns = np.zeros(self.filter_len)
        if (do_log):
            modelLog = open('log/multiEpisode.log', 'w')
            modelLog.write("Arranca run_multiple_episode" + '\n')
        for i in range(0, self.filter_len):
            returns[i] = self.run_one_episode(False,indiv,'RS')
            if do_log:
                modelLog.write("retorno: %s" %(returns[i]) + '\n')
        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]
        if do_log:
            modelLog.write("Worst cases mean: %s, Total mean: %s" % (np.mean(worst_cases), np.mean(returns)) + '\n')
        return [np.mean(worst_cases), np.mean(returns)]




    def optimizeHyperOpt(self):
        starttime = datetime.datetime.now()
        ba.hyperOptInitialize(self,self.twModel.name)
        ba.hyperOptBayesOptim()
        elapsedTime = datetime.datetime.now() - starttime
        print('elapsed time: %s'%(elapsedTime))
        ba.setRootPath()
        ba.save_model()
        ba.plot_result()

    def optimizePYGAD(self):
        starttime = datetime.datetime.now()
        ga.initialize(self,self.twModel.name)
        #ga.printGAPopulation()
        ga.run()
        elapsedTime = datetime.datetime.now() - starttime
        print('elapsed time: %s'%(elapsedTime))
        ga.setRootPath()
        ga.printReport()
        ga.save_config()
        ga.save_model()
        ga.plot_result()
        #ga.save()



    def optimize(self, ts=datetime.timedelta(seconds=60), max_steps=100000):
        # Break symmetry by adding noise
        self.twModel.addNoise('Weight',0.5,15)
        self.twModel.addNoise('Vleak',8,8)
        self.twModel.addNoise('Gleak',0.2,8)
        self.twModel.addNoise('Sigma',0.2,10)
        self.twModel.addNoise('Cm',0.1,10)
        self.twModel.commitNoise()

        r_values = np.zeros(max_steps+1)
        r_counter = 0


        #return [np.mean(worst_cases), np.mean(returns)]
        (current_return, mean_ret) = self.run_multiple_episodes()
        r_values[r_counter] = mean_ret
        r_counter += 1

        #Ariel
        theBestRewardOfWorstCases=current_return
        theBestRewardAvg=0
        haveAnyDump=False


        num_distortions = 6
        num_distortions_sigma = 5
        num_distortions_vleak = 5
        num_distortions_gleak = 4
        num_distortions_cm = 4
        steps_since_last_improvement = 0
        steps_since_last_improvement2  = 0
        starttime = datetime.datetime.now()
        #endtime = starttime + ts
        steps = 0
        log_freq = 10
        #while endtime > datetime.datetime.now() and steps < max_steps:
        while steps < max_steps:
            print('#%s'%(steps))
            steps += 1

            # weight
            distortions = rng.randint(0, num_distortions)
            variance = rng.uniform(0.01, 0.8)

            # sigma
            distortions_sigma = rng.randint(0, num_distortions_sigma)
            variance_sigma = rng.uniform(0.01, 0.08)

            # vleak
            distortions_vleak = rng.randint(0, num_distortions_vleak)
            variance_vleak = rng.uniform(0.1, 3)

            # vleak
            distortions_gleak = rng.randint(0, num_distortions_gleak)
            variance_gleak = rng.uniform(0.05, 0.8)

            # cm
            distortions_cm = rng.randint(0, num_distortions_cm)
            variance_cm = rng.uniform(0.01, 0.3)

            self.twModel.addNoise('Weight', variance, distortions)
            self.twModel.addNoise('Sigma', variance_sigma, distortions_sigma)
            self.twModel.addNoise('Vleak',variance_vleak,distortions_vleak)
            self.twModel.addNoise('Cm',variance_cm,distortions_cm)
            self.twModel.addNoise('Gleak',variance_gleak,distortions_gleak)

            (new_return, mean_ret) = self.run_multiple_episodes()
            #print(str(new_return))
            r_values[r_counter] = mean_ret
            r_counter += 1
            # print('Stochastic Return: '+str(new_return))
            if (new_return > current_return):
                print('Improvement! New Return: %s, old: %s' % (new_return,current_return))
                if new_return>theBestRewardOfWorstCases and new_return>90:
                    #avg = self.evaluate_avg(10)
                    avg = new_return
                    #if(avg>theBestRewardAvg):
                    print('Improvement! New Return: %s, old: %s, new avg: %s' % (new_return,current_return,avg))
                        #theBestRewardAvg=avg
                    self.twModel.dumpModel(self.rootPath+'/dumpBestResult_'+str(new_return)+'_'+str(avg)+'.xml')
                    theBestRewardOfWorstCases=new_return
                    haveAnyDump=True
                if (self.logfile != None):
                    elapsed = datetime.datetime.now() - starttime
                    self.logfile.write('Improvement after: ' + str(steps) + ' steps, with return ' + str(
                        new_return) + ', Elapsed: ' + str(elapsed.total_seconds()) + '\n')
                    self.logfile.flush()

                current_return = new_return
                self.twModel.commitNoise()
                steps_since_last_improvement = 0
                steps_since_last_improvement2 = 0

                num_distortions -= 1
                if (num_distortions < 6):
                    num_distortions = 6

                num_distortions_sigma -= 1
                if (num_distortions_sigma < 5):
                    num_distortions_sigma = 5

                num_distortions_vleak -= 1
                if (num_distortions_vleak < 4):
                    num_distortions_vleak = 4

                num_distortions_gleak -= 1
                if (num_distortions_gleak < 4):
                    num_distortions_gleak = 4

                num_distortions_cm -= 1
                if (num_distortions_cm < 4):
                    num_distortions_cm = 4
                # print('Set Distortion to '+str(num_distortions))
            else:
                steps_since_last_improvement += 1
                steps_since_last_improvement2 +=1
                self.twModel.revertNoise()


                # no improvement seen for 100 steps
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 < 300):
                    print('more than 50 steps...')
                    steps_since_last_improvement = 0
                    steps_since_last_improvement2 += 1

                    # reevaluate return
                 #Esto lo comente por que me parecio sin sentido asi coinciden ademas lor rcounter
                 #   #(current_return, mean_ret) = self.run_multiple_episodes()
                 #   #r_values[r_counter] = mean_ret
                 #   #r_counter += 1
                    # print('Reevaluate to: '+str(current_return))
                    if (self.logfile != None):
                        self.logfile.write(
                            'Reevaluate after: ' + str(steps) + ' steps, with return ' + str(new_return) + '\n')
                        self.logfile.flush()

                    # Increase variance
                    num_distortions += 1
                    if (num_distortions > 16):
                        num_distortions = 12
                    # Increase variance sigma
                    num_distortions_sigma += 1
                    if (num_distortions_sigma > 12):
                        num_distortions_sigma = 12
                    # Increase variance vleak
                    num_distortions_vleak += 1
                    if (num_distortions_vleak > 8):
                        num_distortions_vleak = 8
                    # Increase variance vleak
                    num_distortions_gleak += 1
                    if (num_distortions_gleak > 8):
                        num_distortions_gleak = 8
                    # Increase variance cm
                    num_distortions_cm += 1
                    if (num_distortions_cm > 7):
                        num_distortions_cm = 7
                if (steps_since_last_improvement > 50)and(steps_since_last_improvement2 > 300):
                    print('more than 300 steps without improvement...')
                    steps_since_last_improvement=0
                    steps_since_last_improvement2=0
                    if (self.logfile != None):
                        self.logfile.write(
                            'Much Noise after 300 steps...\n')
                        self.logfile.flush()

                    self.twModel.addNoise('Weight', 0.5, 26)
                    self.twModel.addNoise('Vleak', 10, 11)
                    self.twModel.addNoise('Gleak', 0.2, 11)
                    self.twModel.addNoise('Sigma', 0.2, 26)
                    self.twModel.addNoise('Cm', 0.1, 11)
                    self.twModel.commitNoise()

            if (steps % log_freq == 0 and self.csvlogfile != None):
                elapsed = datetime.datetime.now() - starttime
                #avg_rew = self.evaluate_avg()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(str(steps) + ';' + str(new_return) + ';' + str(performance_r) + ';' + str(
                    elapsed.total_seconds()) + '\n')
                self.csvlogfile.flush()
                # outfile = logdir+'/tw-'+str(worker_id)+'_steps-'+str(steps)+'.bnn'
                # lif.WriteToFile(outfile)
                # print('Set Distortion to '+str(num_distortions))
        if (self.logfile != None):
            self.logfile.write('Total steps done: ' + str(steps) + '\n')
            self.logfile.close()
        if (self.csvlogfile != None):
            elapsed = datetime.datetime.now() - starttime
            #avg_cost = self.evaluate_avg()
            performance_r = np.mean(r_values[0:r_counter])
            self.csvlogfile.write(
                str(steps) + ';' + str(new_return) + ';' + str(performance_r) + ';' + str(elapsed.total_seconds()) + '\n')
            self.csvlogfile.flush()
        if not haveAnyDump:
            self.twModel.dumpModel(self.rootPath + '/dumpBestResult.' + str(new_return) + '.xml')
        print('rconter final %s'%(r_counter))
        print('elapsed %s' %(datetime.datetime.now() - starttime))

    def generatePlots(self):
        with open(self.rootPath+'/csvlog_0.log') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            stepsList=[]
            currentReturn=[]
            currentMeanReturn=[]
            elapsedTime=[]
            for row in csv_reader:
                stepsList.append(float(row[0]))
                currentReturn.append(float(row[1]))
                currentMeanReturn.append(float(row[2]))
                elapsedTime.append(float(row[3]))

        #print(stepsList)
        #print(currentReturn)
        #print(currentMeanReturn)
        #print(elapsedTime)
        plt.plot(stepsList, currentReturn, 'ro',label='current reward')
        plt.plot(stepsList,currentMeanReturn,'bo',label='average reward')
        plt.ylabel('reward')
        plt.xlabel('optimization steps')
        plt.legend()
        plt.title('Rewards vs optimization steps')
        plt.savefig(self.rootPath+'/rewaqrds.png', bbox_inches='tight')
        plt.show()


    def replay(self):
        #self.load_tw(filename)
        if not os.path.exists(self.rootPath+'/vid'):
            os.makedirs(self.rootPath+'/vid')
        replaylog = open(self.rootPath+'/replay.txt', 'w')
        eval=self.evaluate_avg()
        #multipleResult =self.run_multiple_episodes()
        replaylog.write('Mean Reward, varianza, desv standar: ' + str(eval)+'\n')
        #replaylog.write('Replay Return: ' + str(multipleResult)+'\n')
        print('Average Reward: ' + str(eval))
        #print('Replay Return: ' + str(multipleResult))
        replaylog.close()
        #self.run_one_episode(True)

    def generateVideo(self):
        path = self.rootPath + '/vid/'
        command = "ffmpeg -framerate 24 -i " + path + "img_%05d.png " + path + "output.mp4"
        result=os.system(command)
        if result != 0:
            print('Error al generar video')
        else:
            print('Video generated correctly')
            command = "rm -rf "+path + "*.png"
            result = os.system(command)
            if result==0:
                print('Correctamente borrado de .png files')
            else:
                print('Error en borrado de .png files con code: %s' %(result))


    def replay_arg(self):

        worker_id = 1
        if (len(sys.argv) > 1):
            worker_id = int(sys.argv[1])

        filename = 'bnn1/tw-optimized_' + str(worker_id) + '.bnn'
        self.load_tw(filename)

        print('Replay Return: ' + str(self.run_multiple_episodes()))

        self.run_one_episode(True)

    def optimize_and_store(self, worker_id, in_file='tw_pure.bnn'):
        #self.load_tw(in_file)


        #twModel.Reset()
        #if (worker_id.isdigit()):
        #    seed = int(worker_id) + 20 * datetime.datetime.now().microsecond + 23115
        #else:
        #    seed = 20 * datetime.datetime.now().microsecond + 23115

        #self.lif.SeedRandomNumberGenerator(seed)
        rng.seed(423)
        log_path = self.rootPath
        log_path_txt = self.rootPath
        store_path = self.rootPath

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(log_path_txt):
            os.makedirs(log_path_txt)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        log_file = self.rootPath + '/textlog_' + worker_id + '.log'
        csv_log = self.rootPath + '/csvlog_' + worker_id + '.log'
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')

        print('Begin Return of ' + worker_id + ': ' + str(self.run_multiple_episodes()))
        #self.optimize(ts=datetime.timedelta(hours=2), max_steps=100)
        self.optimize(ts=datetime.timedelta(hours=2), max_steps=self.optimSteps)
        #self.optimize(ts=datetime.timedelta(hours=12), max_steps=50000)
        print('End Return: of ' + worker_id + ': ' + str(self.run_multiple_episodes()))

        #outfile = store_path + '/tw-optimized_' + worker_id + '.bnn'
        #self.twModel.dumpModel('dumpTWModel.xml')
        #self.lif.WriteToFile(outfile)


def demo_run():
    env = gym.make("MountainCarContinuous-v0")
    #print('Observation space: '+str(env.observation_space.shape[0]))
    #print('Action space: '+str(env.action_space.shape[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument('--filter', default=1, type=int)
    parser.add_argument('--mean', default=1, type=int)
    parser.add_argument('--file', default="tw_pure.bnn")
    parser.add_argument('--command', default='', type=str)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    twenv = TWsearchEnv(env, args.filter, args.mean,args.steps)

    #twenv.twModel.load('')dumpBestResult_77.36162198835174

    optimized=twenv.rootPath+'/dumpBestResult_97.7792890098983_97.7792890098983.xml'
    if (args.command=='Optimize'):
        print("Optimize")
        twenv.twModel.load('Letchner/TWLetchBase.xml')
        twenv.twModel.name = 'RandomSeek'
        twenv.optimize_and_store(str(args.id), args.file)
    elif (args.command == 'OptimizeGA'):
        print("OptimizeGA")
        twenv.twModel.load('Letchner/TWLetchBase.xml')
        twenv.twModel.name = 'GeneticAlgor'
        # twenv.twModel.load('results/fil_1_mean_1_steps_1000/dumpBestResult.99.09012119121958.xml')
        twenv.optimizePYGAD()
    elif (args.command == 'OptimizeBO'):
        print("OptimizeBO")
        twenv.twModel.load('Letchner/TWLetchBase.xml')
        twenv.twModel.name = 'BayesOptim'
        # twenv.twModel.load('results/fil_1_mean_1_steps_1000/dumpBestResult.99.09012119121958.xml')
        twenv.optimizeHyperOpt()
    elif (args.command=='Replay'):
        print("Replay")
        twenv.twModel.load(optimized)
        #twenv.twModel.load('results/fil_1_mean_1_steps_1000/dumpBestResult.99.09012119121958.xml')
        twenv.replay()
    elif (args.command == 'ReplayGA'):
        print("ReplayGA")
        twenv.rootPath='Pruebas/98.47000592085854'
        optimized = twenv.rootPath + '/model.xml'
        twenv.twModel.load(optimized)
        # twenv.twModel.load('results/fil_1_mean_1_steps_1000/dumpBestResult.99.09012119121958.xml')
        twenv.replay()
#        twenv.generateVideo()
    elif (args.command == 'ReplayBO'):
        print("ReplayBO")
        twenv.rootPath='BayesOptim/98.88232710490215'
        optimized = twenv.rootPath + '/model.xml'
        twenv.twModel.load(optimized)
        # twenv.twModel.load('results/fil_1_mean_1_steps_1000/dumpBestResult.99.09012119121958.xml')
        twenv.replay()
#        twenv.generateVideo()
    elif (args.command=='Graph'):
        print("Graph")
        twenv.twModel.load(optimized)
        twenv.generatePlots()
    elif (args.command == 'Video'):
        print("Video")
        twenv.twModel.load(optimized)
        twenv.generateVideo()
    else:
        print("You must put a command")
if __name__ == "__main__":
    demo_run()


    #env = gym.make("MountainCarContinuous-v0")
    #print('Observation space: '+str(env.observation_space.low))
    #print('Action space: '+str(env.action_space.low))
    #print(env.reset())
    #print((env.step([1])))
    #print((env.step([1]))[1])
    #print(env.step([1]))

    #twenv = TWsearchEnv(env, 1, 1, 1000)
    #twenv.generatePlots()
    #twenv.twModel.load('Letchner/TWLetchBase.xml')
    #twenv.twModel.load('')
    #print(twenv.twModel)
    #twenv.twModel.dumpModel('Letchner/TWLetchBase2.xml')
    #print(twenv.run_one_episode(False))
    ####----GA----
    #pop = ga.populationCreation()
    #clusteredIndiv = twenv.selectToEvolve(pop)
    #print (clusteredIndiv[0])
    #print('------------------------')
    #print (clusteredIndiv[1])
