# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:20:07 2018

@author: Andrija Master
"""

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from src.models.Nestrukturni import Nestrukturni_fun
from src.models.utils.Nestrukturni_GCRF import Nestrukturni_fun_GCRF
from Struktura import Struktura_fun 
from Struktura_GCRF import Struktura_fun_GCRF
from GCRFCNB import GCRFCNB
from GCRFC import GCRFC
from GCRFC_fast import GCRFC_fast
from GCRF import GCRF
from src.preprocess.Putevi_dataset import output,atribute
from sklearn.model_selection import train_test_split





""" Racunanje """
No_class = 18
NoGraph = 4
ModelUNNo = 4
testsize2 = 0.2
broj_fold = 10
iteracija = 400

file = open("rezultati2.txt", "w")

AUCNB = np.zeros(broj_fold)
AUCB = np.zeros(broj_fold)
AUCBF = np.zeros(broj_fold)
R2GCRF = np.zeros(broj_fold)
MSEGCRF = np.zeros(broj_fold)

logProbNB = np.zeros(broj_fold)
logProbB = np.zeros(broj_fold)
logProbBF = np.zeros(broj_fold)

timeNB = np.zeros(broj_fold)
timeB = np.zeros(broj_fold)
timeBF = np.zeros(broj_fold)
timeGCRF = np.zeros(broj_fold)

Skor_com_AUC = np.zeros([broj_fold,ModelUNNo])
Skor_com_AUC2 = np.zeros([broj_fold,ModelUNNo])

Skor_com_R2 = np.zeros([broj_fold,ModelUNNo])
Skor_R2mean = np.zeros([ModelUNNo])
Skor_com_R22 = np.zeros([broj_fold,ModelUNNo])
Skor_R22mean = np.zeros([ModelUNNo])
Skor_com_MSE = np.zeros([broj_fold,ModelUNNo])
Skor_MSEmean = np.zeros([ModelUNNo])

    
skf = KFold(n_splits = broj_fold)
skf.get_n_splits(atribute, output)

output1 = output.iloc[:,:18]
i = 0

    
x_train_com, x_test, y_train_com, Y_test1 = train_test_split(atribute, output, test_size = 0.2, random_state =31)
x_train_un, x_train_st, y_train_un1, Y_train1 = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
Y_test = Y_test1.iloc[:,:18]
Y_train = Y_train1.iloc[:,:18]
y_train_un = y_train_un1.iloc[:,:18]

Y_test_reg = Y_test1.iloc[:,18:]
Y_train_reg = Y_train1.iloc[:,18:]
y_train_un_reg = y_train_un1.iloc[:,18:]

Skor_com_AUC[i,:], Skor_com_AUC2[i,:], R_train, R_test, R2, Noinst_train, Noinst_test = Nestrukturni_fun(x_train_un, y_train_un, x_train_st, Y_train, x_test, Y_test, No_class)
Se_train, Se_test = Struktura_fun(No_class,NoGraph, R2 , y_train_com, Noinst_train, Noinst_test)    

""" Model GCRFC """

Y_train = Y_train.values
Y_test = Y_test.values
start_time = time.time()
mod1 = GCRFCNB()
mod1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 6e-4, maxiter = iteracija)  
#mod1.alfa = np.array([1-10, 1e-10, 1e-10, 3000])
#mod1.beta = np.array([1.0000000e-10, 1.0000000e-10, 1e-10, 1e-10])
probNB, YNB = mod1.predict(R_test,Se_test)
timeNB[i] = time.time() - start_time


start_time = time.time()
mod2 = GCRFC()
#x0 = np.load('mod2.npy')
mod2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija)  
np.save('mod2',mod2.x)  
#mod1.alfa = np.array([7.67362291, 4.7631527 , 9.79830104])
#mod1.beta = np.array([ 7.01829973, 16.59090051, 18.9508093 ,  5.79445323])
probB, YB, VarB = mod2.predict(R_test,Se_test)
timeB[i] = time.time() - start_time 


start_time = time.time()
mod3 = GCRFC_fast()
#x0 = np.load('mod3.npy')
mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 50)  
np.save('mod3',mod3.x)
#mod1.alfa = np.array([0.1043126 , 0.06905401, 0.08689079])
#mod1.beta = np.array([1.00008728e-08, 2.88191498e+02, 1.00000563e-08, 1.00000000e-08, 8.74943190e+01, 3.48984028e-03])  
probBF, YBF, VarBF = mod3.predict(R_test,Se_test)  
timeBF[i] = time.time() - start_time

""" Model GCRF """


Skor_com_MSE[i,:], Skor_com_R2[i,:], Skor_com_R22[i,:], R_train_reg, R_test_reg, R2_reg, Noinst_train, Noinst_test, Y_train_reg, Y_test_reg = Nestrukturni_fun_GCRF(x_train_un, y_train_un_reg, x_train_st, Y_train_reg, x_test, Y_test_reg, No_class)
Se_train_reg, Se_test_reg = Struktura_fun_GCRF(No_class,NoGraph, R2 , y_train_com, Noinst_train, Noinst_test, Y_train, Y_test)


start_time = time.time()
mod4 = GCRF()
#x0 = np.load('mod4.npy')
mod4.fit(R_train_reg, Se_train_reg, Y_train_reg, learn = 'TNC', maxiter = 5000)  
#    np.save('mod4',mod4.x)
#mod1.alfa = np.array([0.1043126 , 0.06905401, 0.08689079])
#mod1.beta = np.array([1.00008728e-08, 2.88191498e+02, 1.00000563e-08, 1.00000000e-08, 8.74943190e+01, 3.48984028e-03])  
YGCRF = mod4.predict(R_test_reg, Se_test_reg)  
timeGCRF[i] = time.time() - start_time


Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
Y_test_reg1 = Y_test_reg.copy()
Y_test_reg = Y_test_reg.reshape([Y_test_reg.shape[0]*Y_test_reg.shape[1]])
YGCRF1 = YGCRF.copy()
YGCRF = YGCRF.reshape([YGCRF.shape[0]*YGCRF.shape[1]])

YNB =  YNB.reshape([YNB.shape[0]*YNB.shape[1]])
probNB = probNB.reshape([probNB.shape[0]*probNB.shape[1]])
YB =  YB.reshape([YB.shape[0]*YB.shape[1]])
probB = probB.reshape([probB.shape[0]*probB.shape[1]])
YBF =  YBF.reshape([YBF.shape[0]*YBF.shape[1]])
probBF = probBF.reshape([probBF.shape[0]*probBF.shape[1]])

probNB[Y_test==0] = 1 - probNB[Y_test==0]
probB[Y_test==0] = 1 - probNB[Y_test==0]
probBF[Y_test==0] = 1 - probBF[Y_test==0]

AUCNB[i] = roc_auc_score(Y_test,probNB)
AUCB[i] = roc_auc_score(Y_test,probB)
AUCBF[i] = roc_auc_score(Y_test,probBF)
R2GCRF[i] = r2_score(Y_test_reg,YGCRF)
MSEGCRF[i] = mean_squared_error(Y_test_reg,YGCRF)

logProbNB[i] = np.sum(np.log(probNB))
logProbB[i] = np.sum(np.log(probB))
logProbBF[i] = np.sum(np.log(probBF))

file.write('AUC GCRFCNB prediktora je {}'.format(AUCNB[i]) + "\n")
file.write('AUC GCRFCB prediktora je {}'.format(AUCB[i]) + "\n")
file.write('AUC GCRFCB_fast prediktora je {}'.format(AUCBF[i]) + "\n")
file.write('R2 GCRF prediktora je {}'.format(R2GCRF[i]) + "\n")
file.write('R2 GCRF prediktora je {}'.format(MSEGCRF[i]) + "\n")

file.write('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC[i,:]) + "\n")
file.write('AUC2 nestruktuiranih prediktora je {}'.format(Skor_com_AUC2[i,:]) + "\n")
file.write('R2 nestruktuiranih prediktora je {}'.format(Skor_com_R2[i,:]) + "\n")
file.write('R22 nestruktuiranih prediktora je {}'.format(Skor_com_R22[i,:]) + "\n")
file.write('MSE nestruktuiranih prediktora je {}'.format(Skor_com_MSE[i,:]) + "\n")

file.write('Logprob GCRFCNB je {}'.format(logProbNB[i]) + "\n") 
file.write('Logprob GCRFCB je {}'.format(logProbB[i]) + "\n")
file.write('Logprob GCRFCB_fast je {}'.format(logProbBF[i]) + "\n")

file.write("--- %s seconds --- GCRFCNB" % (timeNB[i]) + "\n")
file.write("--- %s seconds --- GCRFCB" % (timeB[i]) + "\n")
file.write("--- %s seconds --- GCRFCB_fast" % (timeBF[i]) + "\n")
file.write("--- %s seconds --- GCRF" % (timeGCRF[i]) + "\n")

file.close()

np.save('Y_test', Y_test_reg1)
np.save('Y_GCRF', YGCRF1)

    