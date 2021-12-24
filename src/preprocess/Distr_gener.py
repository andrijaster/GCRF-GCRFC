# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:42:49 2018

@author: Andrija Master
"""
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sb


df = pd.read_csv("df_slobodno.csv", header = 0, index_col = 0)

atribute = df.iloc[:,:-36]
output = df.iloc[:,-36:]

brzina_NIS = output.loc[:,"('brlabels', '0')"]
brzina_NIS.index = pd.to_datetime(brzina_NIS.index,dayfirst=True)
#brzina_NIS = brzina_NIS[brzina_NIS!=0]
figure = plt.figure(figsize=(13, 6))
ax1 = plt.subplot(211)
ax1.hist(brzina_NIS, bins = 15)
ax1.grid()
ax1.set_xlabel('Prosečna brzina [m/s]')
#ax1.set_xlim(brzina_NIS.index[0],brzina_NIS.index[-1])
#ax1.set_ylim(50,150)
#ax1.set_ylabel('$\mathbf{\lambda}$ [1/min]')
ax2 = plt.subplot(212)
sb.kdeplot(brzina_NIS)
ax2.grid()
ax2.set_xlabel('Prosečna brzina [m/s]')
ax2.get_legend().remove()

brzina_mod = 125.1
kilometraza = pd.read_csv('kilometraza_nis.csv',sep=';')
bg_vreme = kilometraza.iloc[:,0] / brzina_mod
Y_test = np.load('Y_test_reg.npy')
Y_test = Y_test[:,0]
no_test = np.where(Y_test>0)
Y_test = Y_test[no_test]
Y_GCRF = np.load('Y_GCRF.npy')
Y_GCRF = Y_GCRF[:,0]
Y_GCRF = Y_GCRF[no_test]
vreme_test_nis = np.zeros([Y_GCRF.shape[0],bg_vreme.shape[0]])
vreme_razlika = np.zeros([Y_GCRF.shape[0],bg_vreme.shape[0]])
vreme_GCRF_nis = np.zeros([Y_GCRF.shape[0],bg_vreme.shape[0]])
vreme_GCRF_razlika = np.zeros([Y_GCRF.shape[0],bg_vreme.shape[0]])
root_mean_squared = np.zeros([kilometraza.shape[0]])

for i in range(kilometraza.shape[0]):
    vreme_test_nis[:,i] = kilometraza.iloc[i,0]/ Y_test
    vreme_razlika[:,i] = (vreme_test_nis[:,i] - bg_vreme[i])
    vreme_GCRF_nis[:,i] = kilometraza.iloc[i,0]/ Y_GCRF
    vreme_GCRF_razlika[:,i] = (vreme_GCRF_nis[:,i] - bg_vreme[i])
    root_mean_squared[i] = np.sqrt(mean_squared_error(vreme_razlika[:,i], vreme_GCRF_razlika[:,i]))

vreme_test_nis = vreme_test_nis*60
vreme_GCRF_nis = vreme_GCRF_nis*60
vreme_razlika = vreme_razlika*60
vreme_GCRF_razlika = vreme_GCRF_razlika*60
root_mean_squared = root_mean_squared*60