# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:16:34 2017

@author: Weiqing
"""

from sas import SAS
import numpy as np
import math
import matplotlib.pyplot as plt


base = np.linspace(0,math.pi,40)
mat_b = np.array([[math.sin(x) for x in base],
                  [math.cos(x) for x in base],
                  [math.sinh(x) for x in base],
                  [math.cosh(x) for x in base],
                  base])

mat_a = np.random.rand(2000,5)

complete_dat = np.dot(mat_a,mat_b)+np.random.rand(2000,40)
test_dat = complete_dat.copy()
test_dat.ravel()[np.random.choice(test_dat.size,16000,replace=False)]=np.nan

recover = SAS(thres=0.00001,rank = 5)
sasres_a, sasres_b = recover.fit(test_dat,complete_dat,True)

sasres = np.dot(sasres_a.transpose(),sasres_b)
print('MAE of estimation:',np.mean(abs(sasres-complete_dat)[np.isnan(test_dat)]))

fig, ax = plt.subplots()
plt.plot(test_dat[0],'bs',label = 'Data with Missing Entries')
plt.plot(complete_dat[0],'g',label = 'Complete Data',linewidth = 3.0)
plt.plot(sasres[0],'r--', label = 'SAS Recovered Curve')
ax.legend(loc='lower center', shadow=True)
plt.show()