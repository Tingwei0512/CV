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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class Sensor:
  # 最大電池容量: 5焦耳
  MaxBattery=5
  SensorWidth=np.pi/2
  
  def __init__(self, x, y, state, r, ang):
    self.state = state# 0 = 休眠, 1 = 低用電期, 2 = 高用電期
    self.x=x
    self.y=y
    if(r>0):
      self.r=r
      self.ang=ang
    else:
      self.r=np.math.sqrt(x**2 + y**2)
      self.ang=np.math.atan2(y, x)
    while(self.ang<=0):
      self.ang+=np.pi*2
    while(self.ang>np.pi*2):
      self.ang-=np.pi*2
    # 假設初始電量為 隨機
    self.capacity=np.random.uniform(low=0.0, high=Sensor.MaxBattery)
    # 假設接收範圍角度為 90度
    # self.SensorWidth=np.pi/2
    self.charging_order_list = []
  # 充入receive焦耳的能量給sensor
  # 回傳實際充電而消耗的能量
  def charge(self, receive):
    if(self.capacity == Sensor.MaxBattery):
      return 0
    old=self.capacity
    self.capacity+=receive
    if(self.capacity>Sensor.MaxBattery*0.95):
      power_consumed=self.capacity-old
      self.capacity=Sensor.MaxBattery
      return power_consumed
    power_consumed=self.capacity-old
    return power_consumed
  

  def hbhc(self, basic_consume):
    if(self.state == 0):
      self.capacity -= 0.1 * (basic_consume / (self.capacity))
    elif(self.state == 1):
      self.capacity -= basic_consume / (self.capacity)
    else:
      self.capacity -= 2 * (basic_consume / (self.capacity))
    if(self.capacity < 0):
      self.capacity = 0.01

  def hblc(self, basic_consume):

    if(self.state == 0):
      self.capacity -= 0.1 * (basic_consume * self.capacity)
    elif(self.state == 1):
      self.capacity -= basic_consume * self.capacity
    else:
      self.capacity -= 2 * (basic_consume * self.capacity)
    if(self.capacity < 0):
      self.capacity = 0.01

  def fix(self, basic_consume):

    if(self.state == 0):
      self.capacity -= 0.1 * basic_consume
    elif(self.state == 1):
      self.capacity -= basic_consume
    else:
      self.capacity -= 2 * basic_consume
    if(self.capacity < 0):
      self.capacity = 0.01

  def ran(self, basic_consume):

    if(self.state == 0):
      self.capacity -= 0.1 * (basic_consume * random.random()*2)
    elif(self.state == 1):
      self.capacity -= basic_consume * random.random()*2
    else:
      self.capacity -= 2 * (basic_consume * random.random()*2)
    if(self.capacity < 0):
      self.capacity = 0.01


    

class PowerStation:
  # 0~2pi
  BeamWidth=np.pi/4#範圍90度:np.pi/4；範圍45度:np.pi/8;範圍30度:np.pi/12;範圍60度:np.pi/6
  MaxPower=5 # 焦耳
  # Pass Loss Conecrned
  alpha=3

  def __init__(self, value, x=0.0, y=0.0, who=None):
    # 假設在原點
    self.x=x
    self.y=y
    self.setDir(value)
    if(who is not None):
      self.who=who

  def setDir(self, value):
    while(value<=0):
      value+=np.pi*2
    while(value>np.pi*2):
      value-=np.pi*2
    self.Direction=value
    self.anglo=self.Direction-PowerStation.BeamWidth
    self.anghi=self.Direction+PowerStation.BeamWidth
    while(self.anglo<=0):
      self.anglo+=(np.pi*2)
    while(self.anghi>2*np.pi):
      self.anghi-=(np.pi*2)
  
  # copy a power station to simulate
  def Copy(self):
    return PowerStation(self.Direction, x=self.x, y=self.y)

  def in_range(self, sensor):
    sensor_x = sensor.x - self.x
    sensor_y = sensor.y - self.y
    sensor_r, sensor_ang = cart2pol(sensor_x, sensor_y)
    while(sensor_ang<=0):
      sensor_ang+=np.pi*2
    while(sensor_ang>np.pi*2):
      sensor_ang-=np.pi*2
    if(self.anghi > self.anglo):
      if(sensor_ang <= self.anghi and sensor_ang >= self.anglo):
        return True
    else:
      if(sensor_ang <= self.anghi or sensor_ang >= self.anglo):
        return True
    return False

  def transmit(self, sensor, pass_lossed_power):
    if(self.in_range(sensor)):
      """
      if(pass_lossed_power > PowerStation.MaxPower):
        pass_lossed_power=PowerStation.MaxPower
      """
      return sensor.charge(pass_lossed_power)
    return 0

  def GreedyAngle(self, sensor_list, charing_times, pass_loss_power_list):
    best_ang=0
    max_charging=0
    for greedyang in range(0, 72):
      greedyang*=5
      self.setDir(greedyang*np.pi/180)
      sum = 0
      for i in range(len(sensor_list)):
        if(self.in_range(sensor_list[i])):
          distance=sensor_list[i].r
          old=sensor_list[i].capacity
          for j in range(charing_times):
            pass_lossed_power=pass_loss_power_list[self.who, i]
            sum = sum + sensor_list[i].charge(pass_lossed_power)
          sensor_list[i].capacity=old
      if(sum > max_charging):
        best_ang=greedyang
        max_charging=sum
    return best_ang

class EHModel:

  ChargingTimes=1 # 每個角度充幾次，相當於每充n秒才能換角度

  def __init__(self, power_station_list, sensor_list, pass_loss_power_list = None):
    self.power_station_list=power_station_list
    self.sensor_list=sensor_list
    self.sensor_num=len(sensor_list)
    self.power_station_num=len(power_station_list)
    if pass_loss_power_list is None:
      self.pass_loss_power_list = np.zeros([self.power_station_num, self.sensor_num])
    else:
      self.pass_loss_power_list = pass_loss_power_list

    for i in range(self.sensor_num):
      self.sensor_list[i].charging_order_list = list(range(self.power_station_num))
      self.quicksort(self.sensor_list[i].charging_order_list, self.sensor_list[i], 0, self.power_station_num-1)
      if pass_loss_power_list is None:
        for j in range(self.power_station_num):
          distance=np.math.sqrt((self.power_station_list[j].x-self.sensor_list[i].x)**2+(self.power_station_list[j].y-self.sensor_list[i].y)**2)
          output = PowerStation.MaxPower*(4*(np.pi**2)/(PowerStation.BeamWidth*2*Sensor.SensorWidth))*(distance**(-PowerStation.alpha))
          if(output > PowerStation.MaxPower):
            output = PowerStation.MaxPower
          self.pass_loss_power_list[j, i] = output

  def Copy(self):
    return EHModel(self.power_station_list, copy.deepcopy(self.sensor_list), self.pass_loss_power_list)

  def total_sensor_capacity(self):
    total=0
    for i in self.sensor_list:
      total+=i.capacity
    return total

  def log_total_sensor_capacity(self):
    total=0
    for i in self.sensor_list:
      total+=math.log(i.capacity + 1)
    return total

  def MaxCapacity(self):
    return self.sensor_list[0].MaxBattery*self.sensor_num

  def quicksort(self, data, sensor, left, right):
      if left >= right :            
          return

      i = left                      
      j = right                     
      key = data[left]                

      while i != j:                  
          while (not self.lesseq(data[j], key, sensor)) and i < j:   
              j -= 1
          while self.lesseq(data[i], key, sensor) and i < j: 
              i += 1
          if i < j:                      
              data[i], data[j] = data[j], data[i]

      data[left] = data[i] 
      data[i] = key

      self.quicksort(data, sensor, left, i-1)   
      self.quicksort(data, sensor, i+1, right) 
  def lesseq(self, a, b, sensor):
    dis_a = np.math.sqrt((self.power_station_list[a].x-sensor.y)**2 + (self.power_station_list[a].y-sensor.y)**2)
    dis_b = np.math.sqrt((self.power_station_list[b].x-sensor.y)**2 + (self.power_station_list[b].y-sensor.y)**2)

    return dis_a <= dis_b

  def start(self, solution):
    MaxCapacity=self.MaxCapacity()
    station_directions = np.zeros([self.power_station_num, solution.size//self.power_station_num])

    for time_slice in range(self.power_station_num):
      station_directions[time_slice]=solution[time_slice::self.power_station_num]

    total_time_slot = solution.size//self.power_station_num
    for time_solt in range(total_time_slot):
      i = 0
      for sensor in self.sensor_list:
        charging_order = sensor.charging_order_list
        if sensor in charging_order:
          for charging_times in range(EHModel.ChargingTimes):
            for power_station in charging_order:
              self.power_station_list[power_station].setDir(station_directions[power_station, time_solt])
              self.power_station_list[power_station].transmit(sensor, self.pass_loss_power_list[power_station, i])
          i+=1
        else:
          sensor.hbhc(1) #有hbhc, hblc, fix, rand

        if time_solt % 5 == 0:
          sensor.state = random.randint(0, 2)

    now=float(self.total_sensor_capacity()/MaxCapacity)
    log_now = float(self.log_total_sensor_capacity()/MaxCapacity)
    return log_now * (now / (5*total_time_slot))

  def showup(self, solution, area_range=12, plotting=True):
    area_width=area_range  # 分布區域的寬
    area_height=area_range   # 分布區域的高

    time_count=1
    station_directions = np.zeros([self.power_station_num, solution.size//self.power_station_num])
    charging_order = [] #list(range(self.power_station_num))

    for time_slice in range(self.power_station_num):
      station_directions[time_slice]=solution[time_slice::self.power_station_num]

    total_time_slot = solution.size//self.power_station_num
    for time_solt in range(total_time_slot):
      i = 0
      for sensor in self.sensor_list:
        charging_order = sensor.charging_order_list
        #self.quicksort(charging_order, sensor, 0, self.power_station_num-1)
        for charging_times in range(EHModel.ChargingTimes):
          for power_station in charging_order:
            self.power_station_list[power_station].setDir(station_directions[power_station, time_solt])
            self.power_station_list[power_station].transmit(sensor, self.pass_loss_power_list[power_station, i])
        i+=1
      if(self.total_sensor_capacity() != self.MaxCapacity()):
        time_count+=1

    if(plotting):
      fig=plt.figure(figsize=(4,4))
    
    for i in self.sensor_list:
      alpha=i.capacity/Sensor.MaxBattery
      if(plotting):
        if(alpha==1.0):
          plt.scatter(i.x, i.y, facecolor='r', alpha=float(alpha))
        else:
          plt.scatter(i.x, i.y, facecolor='b', alpha=float(alpha)/2.0+0.3)
    
    if(plotting):
      plt.xticks(range(-area_width//2,area_width//2+1, 1))
      plt.yticks(range(-area_height//2,area_height//2+1, 1))

      circ = plt.Circle((0, 0), radius=area_width/2, color='black', linewidth=1, fill=False)

      ax = plt.gca()
      ax.spines['top'].set_color('none')
      ax.spines['right'].set_color('none')
      ax.spines['bottom'].set_position(('data', 0))
      ax.spines['left'].set_position(('data', 0))

      ax.add_artist(circ)
      plt.show()

    return time_count
  def printSensorCap(self):
    print(50*'=')
    for sensor in self.sensor_list:
      print(sensor.capacity)
    print(50*'=')
  def GreedyAngle(self, which):
    best_ang=0
    max_charging=0
    for greedyang in range(0, 24):
      greedyang*=15
      self.power_station_list[which].setDir(greedyang*np.pi/180)
      sum = 0.0
      for i in range(self.sensor_num):
        if(self.power_station_list[which].in_range(self.sensor_list[i])):
          old=self.sensor_list[i].capacity
          for j in range(EHModel.ChargingTimes):
            pass_lossed_power=self.pass_loss_power_list[which, i]
            sum += self.sensor_list[i].charge(pass_lossed_power)
          self.sensor_list[i].capacity=old
      #print(greedyang, sum)
      if(sum > max_charging):
        best_ang=greedyang
        max_charging=sum
    #print(best_ang, max_charging)
    return best_ang

  def singleCharging(self, time_slot):
    if(time_slot.size == self.power_station_num):
      i = 0
      for sensor in self.sensor_list:
        charging_order = sensor.charging_order_list
        #self.quicksort(charging_order, sensor, 0, self.power_station_num-1)
        #print("Charging Order:", charging_order)
        for charging_times in range(EHModel.ChargingTimes):
          for power_station in charging_order:
            self.power_station_list[power_station].setDir(time_slot[power_station])
            self.power_station_list[power_station].transmit(sensor, self.pass_loss_power_list[power_station, i])
        i+=1
    else:
      print("wrong")

#num_of_power_station = 1

def distribution(low=0, limit=1 ,size=1 ,dtype="uniform", exponent=None):
  if  dtype == "uniform":
    return ((limit-low)*scipy.stats.uniform.rvs(loc=0, scale=1, size=(size,1))+low, scipy.stats.uniform.rvs(0, 2*np.pi, (size,1)))
  elif dtype == "power_law":
    if exponent is None:
      exponent = 2.5
    s = nx.utils.powerlaw_sequence(size, exponent=exponent)
    G = nx.expected_degree_graph(s, selfloops=False)
    pos = nx.spring_layout(G)
    pos = list(pos.items())
    pos = np.array(pos)
    pos = np.transpose(pos)
    pos = np.array(pos[1])
    pos = np.transpose(pos)
    re = np.zeros([size, 2])
    for i in range(size):
      re[i][0]=pos[i][0]
      re[i][1]=pos[i][1]
    re = np.transpose(re)
    return cart2pol(re[0]*(limit-low)+low, re[1]*(limit-low)+low)
  elif dtype == "BA" or dtype =="ba":
    if exponent is None:
      exponent = 1
    #s = nx.utils.powerlaw_sequence(size, exponent=exponent)
    G = nx.random_graphs.barabasi_albert_graph(size, exponent)
    pos = nx.spring_layout(G)
    pos = list(pos.items())
    pos = np.array(pos)
    pos = np.transpose(pos)
    pos = np.array(pos[1])
    pos = np.transpose(pos)
    re = np.zeros([size, 2])
    for i in range(size):
      re[i][0]=pos[i][0]
      re[i][1]=pos[i][1]
    re = np.transpose(re)
    return cart2pol(re[0]*(limit-low)+low, re[1]*(limit-low)+low)

area_width=30  # 分布區域的寬
area_height=30  # 分布區域的高
density_lambda=1000/(area_height*area_width) # Poisson process的密度
area_total=area_height*area_width #區域面積

# 區域內點數量
total_num=(int)(area_total*density_lambda)

# Poisson point process

#numbPoints = scipy.stats.poisson(total_num).rvs()
numbPoints = 100
sensor = []
#r = (area_width//2)*scipy.stats.uniform.rvs(0, 1, (numbPoints,1))
#r = (area_width//2)*scipy.stats.powerlaw.rvs(a=1.5, loc=0, scale=1, size=(numbPoints, 1))
#r = distribution(low=0, limit=area_height//2, size=numbPoints, dtype="power_law", exponent=2.5)
#ang = scipy.stats.uniform.rvs(0, 2*np.pi, (numbPoints,1))

r1, ang1 = distribution(low=0, limit=area_height/2, size=numbPoints, dtype="uniform")

x1=r1*np.cos(ang1)
y1=r1*np.sin(ang1)


for i in range(numbPoints):
  sensor.append(Sensor(x1[i], y1[i], -1, 0))

r, ang = distribution(low=0, limit=area_height/6.5, size=numbPoints//2, dtype="power_law", exponent=2.5)

x=r*np.cos(ang)+area_height//5
y=r*np.sin(ang)+area_height//5

for i in range(numbPoints//2):
  sensor.append(Sensor(x[i], y[i], -1, 0))

power_station_list = []
print(len(sensor))

#power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=0.0, y=0.0))

x, y = 0, area_height/4
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y, who=0))
x, y = -area_height*3/8/np.math.sqrt(3), -area_height/8
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y, who=1))
x, y = area_height*3/8/np.math.sqrt(3), -area_height/8
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y, who=2))

MyModel=EHModel(power_station_list, sensor) 


# plotting
print("點數量:", MyModel.sensor_num)
#print(MyModel.pass_loss_power_list)
poisson_process=plt.figure(figsize=(10,10))

plt.scatter(x, y, edgecolors='b', facecolor='r', alpha=0.3)
plt.scatter(x1, y1, edgecolors='b', facecolor='r', alpha=0.3)
for i in power_station_list:
  plt.scatter(i.x, i.y, facecolor='g')
for i in sensor:
  for j in power_station_list:
    if(j.in_range(i)):
      plt.scatter(i.x, i.y, edgecolors='r', facecolor='r', alpha=0.8)


plt.xticks(range(-area_width//2,area_width//2+1, 5))

plt.yticks(range(-area_height//2,area_height//2+1, 5))
draw_circle = plt.Circle((0.0, 0.0), area_height, fill=False)

ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

for i in power_station_list:
  PowerStation_Range=matplotlib.patches.Wedge((i.x, i.y), area_height, 
                        theta1=i.anglo*180/np.pi, theta2=i.anghi*180/np.pi,
                        alpha=0.2, color='blue')
  ax.add_patch(PowerStation_Range)

#circ = plt.Circle((0, 0), radius=area_width/2, color='black', linewidth=1, fill=False)

#ax.add_artist(circ)
#print("MaxCapcity=", MyModel.MaxCapacity())
#ax.xaxis.set_ticks_position('mid')
#plt.xlabel('寬')plt.ylabel('寬')
#plt.legend(loc = 'upper right', fontsize=20)
#plt.savefig("testFig.png")
#plt.show()


time_slot = 90

# GA
RUNNING_TIMES = 100 #總共跑幾次
numbPoints = 20 #sensor
p = 3
d = time_slot # total time slot
desired_output = 0 # Function output.
num_generations = 500 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool
input_function=tf.range(start=d, limit=0, delta=-1)*1000

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




power_station_list = []
sensor_list = []
r, ang = distribution(low=0, limit=area_height/2, size=numbPoints, dtype="ba", exponent=1)

x=r*np.cos(ang)
y=r*np.sin(ang)
save_x = tf.convert_to_tensor(x)
save_y = tf.convert_to_tensor(y)
save = tf.stack([save_x, save_y], axis = 0)
save = tf.squeeze(save)
distribution_data = pd.DataFrame(save.numpy())
distribution_data.to_csv("45度20個sensor使用BA無尺度網路拓樸圖.csv", index=False)

for i in range(numbPoints):
  sensor_list.append(Sensor(x[i], y[i], -1, 0))

x, y = 0, area_height/4
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))
x, y = -area_height*3/8/np.math.sqrt(3), -area_height/8
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))
x, y = area_height*3/8/np.math.sqrt(3), -area_height/8
power_station_list.append(PowerStation(np.random.uniform(low=0.0, high=2*np.pi), x=x, y=y))

MyModel=EHModel(power_station_list, sensor_list)



time_slot = 360
#RR
test = MyModel.Copy()
sol = RR_sol
data2_sensor100_RR =test.showup(solution=RR_sol.numpy(), area_range=area_height, plotting=True)
print("Round Robin:", data2_sensor100_RR)
#Uniform
for times in range(RUNNING_TIMES):
  test = MyModel.Copy()
  sol = tf.random.uniform(shape=[time_slot*test.power_station_num], minval=0, maxval=2*np.pi)
  data2_sensor100_uniform[times]=test.showup(solution=sol.numpy(), area_range=area_height, plotting=False)
  print("Uniform:", times)
print("Uniform:", np.average(data2_sensor100_uniform))

# Greedy
time_slot = 90
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
print("Greedy:", data2_sensor100_greedy)

  # run GA
time_slot = data2_sensor100_greedy+(data2_sensor100_RR//4)*3
for times in range(RUNNING_TIMES):

  def callback_generation(ga_instance):
    global data1_sensor100_GA
    global times
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
    return np.sum(weight*outcome) #+GA_power_station.total_sensor_capacity()*1000

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
              mutation_probability=0.05
              )
  ga_instance.run()
  data2_sensor100_GA[times] = MyModel.Copy().showup(ga_instance.best_solution()[0], area_range=area_height, plotting=False)
  print("In this GA, time = ", data2_sensor100_GA[times])
  if data2_sensor100_GA[times] >= data2_sensor100_greedy:
    for i in range(10):
      print("*"*10)

data2_sensor100_RR_Greedy = np.zeros(shape=[2])

data2_sensor100_RR_Greedy[0] = data2_sensor100_RR
data2_sensor100_RR_Greedy[1] = data2_sensor100_greedy

data_1 = pd.DataFrame(data1_sensor100_GA)
data_2 = pd.DataFrame(data2_sensor100_GA)
data_3 = pd.DataFrame(data2_sensor100_uniform)
data_4 = pd.DataFrame(data2_sensor100_RR_Greedy)
#data_4 = pd.DataFrame(data2_sensor100_RR)
#data_5 = pd.DataFrame(data2_sensor100_greedy)

data_1.to_csv("45度20個sensor, GA在BA無尺度網路迭代500次的Fitness分布.csv", index=False)
data_2.to_csv("45度20個sensor, 用GA在BA無尺度網路跑100次時間分布.csv", index=False)
data_3.to_csv("45度20個sensor, 用Uniform在BA無尺度網路跑100次時間分布.csv", index=False)
data_4.to_csv("45度20個sensor, 用RR, Greedy在BA無尺度網路時間分布.csv")
#data_4.to_csv("10個sensor, 用Round Robin在UniformDistribution跑100次時間分布.csv", index=False)
#data_5.to_csv("10個sensor, 用Greedy在UniformDistribution跑100次時間分布.csv", index=False)
print("Uniform:", np.average(data2_sensor100_uniform))
print("Round Robin:", data2_sensor100_RR)
print("Greedy:", data2_sensor100_greedy)
print("GA:", np.average(data2_sensor100_GA))
