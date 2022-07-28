import configparser

import pygad
import matplotlib.pyplot as plt
import os
import datetime

class GeneticAlgorithm:
    def __init__(self,source=None):
        #params for using pygad
        self.isTimedBased=False
        self.starttime=None
        self.rootpath=''
        self.rlengine=None
        self.fitness_function = None
        self.on_generation = None
        self.initial_population = []
        self.save_best_solutions = True
        self.batch_mode = ''
        self.fit_type = ''
        self.pygadInstance=None
        #params for GA
        self.num_individuos = 0
        self.num_generations = 0
        self.num_parents_mating = 0
        # -1:all, 0:none, n:saved n parents
        self.keep_parents = 0
        # sss: for steady - state selection, rws: for roulette wheel selection, sus: for stochastic universal selection, rank: for rank selection, random: for random selection, and tournament: for tournament selection (add the K_tournament param, default to 3)
        self.parent_selection_type = ""
        self.K_tournament=0
        # single_point: for single - point crossover, two_points: for two points crossover, uniform: for uniform crossover), and scattered: for scattered crossover, if crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations.
        self.crossover_type = ""
        # crossover_probability = None
        # not used
        # random: for random mutation, swap: for swap mutation, inversion: for inversion mutation:, scramble: for scramble mutation, and adaptive: for adaptive mutation. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied
        self.mutation_type = ""
        # mutation_by_replacement = False: se usa si es random la mut, true reemplada, false suma
        self.mutation_by_replacement=True
        #not used
        # si tiene valor no usa el mutation_percent_genes
        #mutation_probability = 0.9
        #mutation_num_genes = 70 or [70,40] for adaptive
        self.mutation_num_genes = None
        #self.mutation_percent_genes = 25
        # [Gleak][Vleak][Cm]
        # [weight][sigma]

        #ask to the model
        self.num_genes=0
        self.gene_type = None
        self.gene_space = None
        self.source=source



    #Our individuals are the solutions so an instance of the value  of the paramters but flattened to a list.
    def populationModelsCreation(self,numIndiv):
        fileNameList = os.listdir(self.source)
        if self.source==None:
            self.rlengine.model.loadFromFile(self.rlengine.getModelBasePath())
            modelToIndiv = self.rlengine.modelInterface.getIndividualForPYGAD(self.rlengine.model)
            self.initial_population.append(modelToIndiv)
            for n in range(1,numIndiv):
                self.rlengine.model.loadFromFile(self.rlengine.getModelBasePath())
                self.rlengine.modelInterface.randonizeModel(self.rlengine.model)
                self.rlengine.model.commitNoise()
                modelToIndiv=self.rlengine.modelInterface.getIndividualForPYGAD(self.rlengine.model)
                self.initial_population.append(modelToIndiv)
        else:
            sourceSize=len(fileNameList)
            if sourceSize<numIndiv:
                for name in fileNameList:
                    file = self.source + '/' + name
                    self.rlengine.restorePickleModel(file)
                    modelToIndiv = self.rlengine.modelInterface.getIndividualForPYGAD(self.rlengine.model)
                    self.initial_population.append(modelToIndiv)
                for n in range(0,numIndiv-sourceSize):
                    self.rlengine.restorePickleModel(self.source+'/baseModel')
                    self.rlengine.modelInterface.randonizeModel(self.rlengine.model)
                    modelToIndiv=self.rlengine.modelInterface.getIndividualForPYGAD(self.rlengine.model)
                    self.initial_population.append(modelToIndiv)
            else:
                for n in range(0,numIndiv):
                    file = self.source + '/' + fileNameList[n]
                    self.rlengine.restorePickleModel(file)
                    modelToIndiv = self.rlengine.modelInterface.getIndividualForPYGAD(self.rlengine.model)
                    self.initial_population.append(modelToIndiv)



    def initialize(self,engine):
        self.starttime = datetime.datetime.now()
        self.fitness_function = self.fitness
        self.on_generation = self.ongen
        self.rootpath = engine.rootpath
        self.rlengine = engine
        self.num_generations = engine.steps
        currDirname = os.path.dirname(__file__)
        patherDir=os.path.join(currDirname, '..')
        path=os.path.join(patherDir,"RLEngine/load.conf")
        config = configparser.RawConfigParser()
        config.read(path)
        self.batch_mode = config.get('GENALG', 'batchmode')
        self.fit_type = config.get('GENALG', 'fit')
        self.num_individuos = int(config.get('GENALG', 'NumIndividuals'))
        self.num_parents_mating = int(config.get('GENALG', 'NumParentsMating'))
        self.keep_parents = int(config.get('GENALG', 'KeepParents'))
        self.parent_selection_type = config.get('GENALG', 'ParentSelectionType')
        self.K_tournament = int(config.get('GENALG', 'KTournament'))
        self.crossover_type=config.get('GENALG', 'CrossoverType')
        self.mutation_type=config.get('GENALG', 'MutationType')
        self.mutation_by_replacement = bool(config.get('GENALG', 'MutationByReplacement'))
        self.mutation_num_genes = parseMutNumGenes(config.get('GENALG', 'MutationNumGenes'))
        self.num_genes=self.rlengine.modelInterface.getNumGenes()
        self.gene_type=self.rlengine.modelInterface.getGeneType()
        self.gene_space=self.rlengine.modelInterface.getGeneSpace()
        #CHEQUEAR
        self.populationModelsCreation(self.num_individuos)
        self.pygadInstance = pygad.GA(num_generations=self.num_generations,
                                          num_parents_mating=self.num_parents_mating,
                                          fitness_func=self.fitness_function,
#                                          callback_generation=self.callback_generation,
                                          num_genes=self.num_genes,
                                          initial_population=self.initial_population,
                                          gene_type=self.gene_type,
                                          parent_selection_type=self.parent_selection_type,
                                          K_tournament=self.K_tournament,
                                          keep_parents=self.keep_parents,
                                          crossover_type=self.crossover_type,
                                          mutation_type=self.mutation_type,
                                          mutation_by_replacement=self.mutation_by_replacement,
                                          mutation_num_genes=self.mutation_num_genes,
                                          gene_space=self.gene_space,
                                          save_best_solutions=self.save_best_solutions,
                                          on_generation=self.on_generation)
                                          #on_fitness=on_fitness,
                                          #on_generation=on_generation,
                                          #on_crossover=on_crossover,
                                          #on_mutation=on_mutation,
                                          #on_parents=on_parents)

    def run(self,mode=None):
        if mode=='timed':
            self.isTimedBased=True
        self.pygadInstance.run()

    def setPath(self):
        label=str(self.pygadInstance.best_solutions_fitness[self.pygadInstance.best_solution_generation])
        if not os.path.exists(self.rootpath + '/' + label):
            os.makedirs(self.rootpath + '/' + label)
        self.rootpath=self.rootpath + '/' + label

    def getFolder(self):
        return str(self.pygadInstance.best_solutions_fitness[self.pygadInstance.best_solution_generation])


    def getPath(self):
        return self.rootpath

    def putOptModelIntoRLEModel(self):
        self.rlengine.modelInterface.putIndividualFromPYGAD(self.rlengine.model, self.pygadInstance.best_solutions[self.pygadInstance.best_solution_generation])

    def plotResult(self,mode=None):
        ordenadas=self.pygadInstance.best_solutions_fitness
        abscisas = [i for i in range(0,len(ordenadas))]
        plt.plot(abscisas, ordenadas, 'ro',label='reward')
        plt.ylabel('reward')
        plt.xlabel('generaciones')
        plt.legend()
        plt.title('Rewards vs Generations')
        plt.savefig(self.rootpath+'/rewards.png', bbox_inches='tight')
        plt.show()

    def saveConfig(self):
        confFile = open(self.rootpath  + '/config.prp', 'w')
        params=self.rlengine.reportParams()
        for key, value in params.items():
            confFile.write(key+'='+str(value)+'\n')
        confFile.write("NumIndividuals=%s\n" % ( self.num_individuos ) )
        confFile.write("NumGenerations=%s\n" % ( self.num_generations ) )
        confFile.write("NumParentsMating=%s\n" % ( self.num_parents_mating ) )
        confFile.write("KTournament=%s\n" % ( self.K_tournament ) )
        confFile.write("ParentSelectionType=%s\n" % ( self.parent_selection_type ) )
        confFile.write("KeepParents=%s\n" % ( self.keep_parents ) )
        confFile.write("CrossoverType=%s\n" % ( self.crossover_type ) )
        confFile.write("MutationType=%s\n" % ( self.mutation_type ) )
        confFile.write("MutationByReplacement=%s\n" % ( self.mutation_by_replacement ) )
        confFile.write("MutationNumGenes=%s\n" % ( self.mutation_num_genes ) )
        confFile.write("fit_type=%s\n" % ( self.fit_type ) )
        confFile.write("batch_mode=%s\n" % ( self.batch_mode ) )
        confFile.close()

    def ongen(self):
        print("Generation", self.pygadInstance.generations_completed)
        if self.isTimedBased:
            elapsedTime = (datetime.datetime.now() - self.starttime).total_seconds()
            if elapsedTime > self.rlengine.steps:
                return "stop"
        # print(ga.population)

    def fitness(self,indiv,indivIdx):
        self.rlengine.modelInterface.putIndividualFromPYGAD(self.rlengine.model,indiv)
        fit = (self.rlengine.runEpisodes(self.fit_type, self.batch_mode))
        #print('fit,',fit)
        # if self.isTimedBased:
        #     elapsedTime = (datetime.datetime.now() - self.starttime).total_seconds()
        #     if elapsedTime > self.rlengine.steps:
        #         self.pygadInstance.run_completed=True
        return fit

    # def callbackgen(self):
    #     if self.isTimedBased:
    #         elapsedTime = (datetime.datetime.now() - self.starttime).total_seconds()
    #         if elapsedTime > self.rlengine.steps:
    #             return "stop"





    def getName(self):
        return 'GeneticAlgorithm'

    def recordResults(self,time):
        bestReward=self.pygadInstance.best_solutions_fitness[self.pygadInstance.best_solution_generation]
        graphicFilesPath='file:///ariel/DataScience/Gusano/BiMPuRLE/RLEngine/'+self.rlengine.rootpath+str(bestReward)
        fileName = self.rlengine.rootpath + '/results.csv'
        file=open(fileName, 'a')
        line = str(self.rlengine.steps) + ',' + str(self.rlengine.batch) + ',' + str(self.rlengine.worst) + ','
        line = line +  self.fit_type+ ','+ self.batch_mode+','+str(self.rlengine.gamma) + ','
        line = line + str(time) + ',' + str(bestReward) +',#,%,@,'+ str(bestReward)
        file.write(line+','+graphicFilesPath+'/rewards.png,'+graphicFilesPath+'/hist.png'+'\n')
        file.close()


def on_parents(ga, selected_parents):
    print("Parents", ga.generations_completed)
    print(selected_parents)


def on_crossover(ga,offspring_crossover):
    print("Crossover", ga.generations_completed)
    print(offspring_crossover)

def on_mutation(ga,offspring_mutation):
    print("Mutation", ga.generations_completed)
    print(offspring_mutation)

def on_fitness(ga,pop_fitness):
    print("Fitness", ga.generations_completed)
    print(pop_fitness)


def parseMutNumGenes(valueFromFile):
    result=None
    strvar=valueFromFile.split(',')
    if (len(strvar)==2):
        result=[]
        result.append(int(strvar[0]))
        result.append(int(strvar[1]))
    else:
        result=int(strvar[0])
    return result




