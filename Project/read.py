import numpy as np
import tensorflow as tf
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import pygad
import copy
import networkx as nx #power_law
import pandas as pd
from Project import distribution, Sensor

def readSystem():
    data = pd.read_csv('./readsample.csv')
    dist = data.to_numpy()
    return dist.transpose()
def newBAGraph(times=0, numbPoints=10, area_height=30, restrict=None):
    graph_flag =True

    if(restrict is None):
        restrict = 0

    while(graph_flag):
        graph_flag=False
        power_station_list = []
        sensor_list = []
        r, ang = distribution(low=0, limit=area_height/2, size=numbPoints, dtype="ba", exponent=1)
        for i in range(numbPoints):
            if(r[i]>area_height/2+restrict):
                graph_flag=True
                break

        if(graph_flag):
            continue

        x=r*np.cos(ang)
        y=r*np.sin(ang)

        for i in range(numbPoints):
            sensor_list.append(Sensor(x[i], y[i], -1, 0))

    save_x = tf.convert_to_tensor(x)
    save_y = tf.convert_to_tensor(y)
    save = tf.stack([save_x, save_y], axis = 0)
    save = tf.squeeze(save)
    distribution_data = pd.DataFrame(save.numpy())
    str1 = "{num}個sensor使用BA無尺度網路拓樸圖_{t}.csv".format(num=numbPoints, t=times)
    distribution_data.to_csv(str1, index=False)
    return sensor_list
