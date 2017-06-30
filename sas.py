# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:07:05 2017

@author: Weiqing
"""

import numpy as np

class SAS:
    def __init__(self,lam=1.5,mu=2,rank=20,thres = 0.001, maxit = 20):
        self.lam = lam
        self.mu = mu
        self.rank = rank
        self.thres = thres
        self.maxit = maxit
        
    def __outersum1(self,indvec,y,lam):
        ytemp = y[:,indvec]
        resmat = np.matmul(ytemp,ytemp.transpose())
        resmat = resmat + np.identity(y.shape[0])*lam
        return resmat
        
    def fit(self,data=None,data1=None,withtrue=False):
        if data==None:
            raise ValueError('data cannot be empty!')
        y = np.random.randn(self.rank,data.shape[1])
        x = np.empty((self.rank,data.shape[0]))
        indmat = ~np.isnan(data)
        
        j = 1
        dist = 1000
        rec = 0
        
        while(j<self.maxit and dist>self.thres):
            for i in range(data.shape[0]):
                x[:,i] = np.matmul(np.linalg.inv(np.matmul(y[:,indmat[i]],y[:,indmat[i]].transpose())+self.lam*np.identity(self.rank)),np.matmul(y[:,indmat[i]],data[i,indmat[i]]))
            y[:,0] = np.matmul(np.linalg.inv(self.__outersum1(indmat[:,0],x,self.lam+self.mu)),(np.matmul(x[:,indmat[:,0]],data[indmat[:,0],0])+self.mu*y[:,1]))
            for i in range(1,y.shape[1]-1):
                y[:,i] = np.matmul(np.linalg.inv(self.__outersum1(indmat[:,i],x,self.lam+2*self.mu)),(np.matmul(x[:,indmat[:,i]],data[indmat[:,i],i])+self.mu*(y[:,i-1]+y[:,i+1])))
            y[:,y.shape[1]-1] = np.matmul(np.linalg.inv(self.__outersum1(indmat[:,y.shape[1]-1],x,self.lam+self.mu)),(np.matmul(x[:,indmat[:,y.shape[1]-1]],data[indmat[:,y.shape[1]-1],y.shape[1]-1])+self.mu*y[:,y.shape[1]-2]))
            
            test = np.matmul(x.transpose(),y)
            dist = abs(rec- np.nansum(abs(test-data))/sum(sum(indmat)))
            rec = np.nansum(abs(test-data))/sum(sum(indmat))
            print('Training Error: ..... ', rec)
            if(withtrue==True):
                testerror = np.nanmean(abs(test - data1)[~indmat])
                print('Testing Error: ...... ',testerror)
            print('Finished Iteration No. ',j)
            j = j+1
        return x, y        
        

