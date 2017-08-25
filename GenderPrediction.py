# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:49:30 2017

@author: vivek
"""
import pandas as pd
from sklearn import tree

X = [[188,88,9],[128,50,7],[230,90,11],[166,55,8],[130,45,7],
     [118,45,6],[156,89,8],[140,62,7],[179,109,9],[90,39,3],[240,110,12]]
Y = ['Male','Female','Male','Male','Female','Female','Female','Male','Male','Female','Male']
DF=pd.DataFrame(X,columns=['Height','Weight','ShoeSize'])
DF['Sex']=Y
print(DF)

DF['SexC']=(DF.Sex=="Female").astype(int)
print(DF)
GR=DF['SexC'].groupby(DF['Height'])
print(GR)
ax1=DF.plot(kind='scatter', x="SexC",y="Height",color='R')
ax2=DF.plot(kind='scatter', x="Weight",y="SexC",color='B')
ax3=DF.plot(kind='scatter', x="ShoeSize",y="SexC",color='G')
print(ax1)
#pd.DataFrame(X,Y).plot(kind='scatter',)
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,76,89]])

print(prediction)