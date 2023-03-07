import numpy as np
import tensorflow as tf
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import pygad
import copy
import networkx as nx #power_law
import pandas as pd
import random, math
from Project import Sensor, PowerStation, distribution, EHModel
from read import readSystem, newBAGraph
area_height = 30

time_slot = 90



# GA
RUNNING_TIMES = 1 #總共跑幾次
numbPoints = 50 #sensor
p = 3
d = time_slot # total time slot
desired_output = 0 # Function output.
num_generations = 1000 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool
input_function=tf.range(start=d, limit=0, delta=-1)

time_slot = 360
pool = tf.range(0, time_slot*p*np.pi, delta=PowerStation.BeamWidth*2)
sol = tf.stack([pool[0:time_slot],pool[0:time_slot],pool[0:time_slot]], axis =0)
sol = tf.transpose(sol)
RR_sol = tf.reshape(sol ,shape=[-1])

data1_sensor100_GA = np.zeros([RUNNING_TIMES, num_generations])
data2_sensor100_GA = np.zeros([RUNNING_TIMES])
data2_sensor100_uniform = np.zeros([RUNNING_TIMES])
#data2_sensor100_RR = np.zeros([RUNNING_TIMES])
#data2_sensor100_greedy = np.zeros([RUNNING_TIMES])


# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 20 # Number of solutions in the population.
num_genes = d*p
times = 0

data2_sensor100_RR = 0
data2_sensor100_greedy = 0

greedy_res = 26
RR_res = 64
flag = True
while(flag):
  power_station_list = []
  sensor_list = []
  distrib = readSystem()
  for i in range(numbPoints):
      sensor_list.append(Sensor(distrib[i][0], distrib[i][1], random.randint(0, 2), -1, 0))

  x, y = 0, area_height/4
  power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))
  x, y = -area_height*3/8/np.math.sqrt(3), -area_height/8
  power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))
  x, y = area_height*3/8/np.math.sqrt(3), -area_height/8
  power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))

  MyModel=EHModel(power_station_list, sensor_list)
  #MyModel.Copy().showup(solution=RR_sol.numpy(), area_range=area_height, plotting=True)

  time_slot = 360
  break
  #
  '''
  #RR
  
  test = MyModel.Copy()
  sol = RR_sol
  data2_sensor100_RR =test.showup(solution=RR_sol.numpy(), area_range=area_height, plotting=False)


  if(data2_sensor100_RR >= RR_res):
    continue

  # Greedy
  time_slot = 150
  test = MyModel.Copy()
  sol = np.zeros([time_slot*test.power_station_num])
  time = time_slot+1

  for i in range(time_slot):
    temp = test.total_sensor_capacity()
    power_station_count=0
    for power_station in test.power_station_list:
      greedyang = test.GreedyAngle(power_station_count)
      power_station.setDir(greedyang*np.pi/180)
      sol[i*test.power_station_num+power_station_count]=greedyang*np.pi/180
      power_station_count+=1
    test.singleCharging(sol[i*test.power_station_num:(i+1)*test.power_station_num])
    if(time>i and test.total_sensor_capacity() == test.MaxCapacity()):
      time = i
  data2_sensor100_greedy = time+1
  

  if(data2_sensor100_greedy >= greedy_res):
    continue
  else:
    flag = False
    
print("Round Robin:", data2_sensor100_RR)
print("Greedy:", data2_sensor100_greedy)
time_slot = 360
#Uniform
for times in range(RUNNING_TIMES):
  test = MyModel.Copy()
  sol = tf.random.uniform(shape=[time_slot*test.power_station_num], minval=0, maxval=2*np.pi)
  data2_sensor100_uniform[times]=test.showup(solution=sol.numpy(), area_range=area_height, plotting=False)
  #print("Uniform:", times)
print("Uniform:", np.average(data2_sensor100_uniform))

'''
  # run GA
time_slot = data2_sensor100_greedy+data2_sensor100_RR//4
for times in range(RUNNING_TIMES):

  def callback_generation(ga_instance):
    global data1_sensor100_GA
    global times
    if(ga_instance.generations_completed%5==0):
      print("Fitness=", ga_instance.best_solution()[1])
      print("{t} - Generation = {generation}".format(t=times, generation=ga_instance.generations_completed))
    data1_sensor100_GA[times][ga_instance.generations_completed-1] = ga_instance.best_solution()[1]

  def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    global input_function
    global MyModel
    sample = MyModel.Copy()
    outcome = sample.start(solution)
    weight = input_function.numpy()
    return outcome #+GA_power_station.total_sensor_capacity()*1000

  fitness_function = fitness_func
  ga_instance = pygad.GA(num_generations=num_generations,
              num_parents_mating=num_parents_mating, 
              #initial_population=[sol, sol, sol, sol,sol, sol, sol, sol,sol, sol, sol, sol,sol, sol, sol, sol,sol, sol, sol, sol],
              #parent_selection_type="random",
              crossover_type="uniform",
              #keep_parents=int(num_parents_mating//4),
              fitness_func=fitness_function,
              #gene_space={'low':0, 'high':np.pi*2+0.01, 'step':PowerStation.BeamWidth},
              sol_per_pop=sol_per_pop,
              #parent_selection_type="sss",
              num_genes=num_genes,
              on_generation=callback_generation,
              #save_best_solutions=True,
              mutation_probability=0.1
              )
  ga_instance.run()
  data2_sensor100_GA[times] = MyModel.Copy().showup(ga_instance.best_solution()[0], area_range=area_height, plotting=False)
  print("In this GA, time = ", data2_sensor100_GA[times])
  print("Gap = ", data2_sensor100_greedy - data2_sensor100_GA[times])

data2_sensor100_RR_Greedy = np.zeros(shape=[2])

data2_sensor100_RR_Greedy[0] = data2_sensor100_RR
data2_sensor100_RR_Greedy[1] = data2_sensor100_greedy

data_1 = pd.DataFrame(data1_sensor100_GA)
data_2 = pd.DataFrame(data2_sensor100_GA)
data_3 = pd.DataFrame(data2_sensor100_uniform)
data_4 = pd.DataFrame(data2_sensor100_RR_Greedy)
#data_4 = pd.DataFrame(data2_sensor100_RR)
#data_5 = pd.DataFrame(data2_sensor100_greedy)

data_1.to_csv("60度50個sensor, GA在Uniform Distribution迭代500次的Fitness分布.csv", index=False)
data_2.to_csv("60度50個sensor, 用GA在Uniform Distribution跑100次時間分布.csv", index=False)
data_3.to_csv("60度50個sensor, 用Uniform在Uniform Distribution跑100次時間分布.csv", index=False)
data_4.to_csv("60度50個sensor, 用RR, Greedy在Uniform Distribution時間分布.csv")
#data_4.to_csv("10個sensor, 用Round Robin在UniformDistribution跑100次時間分布.csv", index=False)
#data_5.to_csv("10個sensor, 用Greedy在UniformDistribution跑100次時間分布.csv", index=False)
print("Uniform:", np.average(data2_sensor100_uniform))
print("Round Robin:", data2_sensor100_RR)
print("Greedy:", data2_sensor100_greedy)
print("GA:", np.average(data2_sensor100_GA))
#2022/3/11==13:50