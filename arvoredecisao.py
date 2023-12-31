# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 22:41:59 2023

@author: Tiago Ventura
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree


class ArvoreDecisao:
   def __init__(self, dataset):
        dataset = pd.read_csv(dataset, header=None)
        
        dataset.columns = ['label'
                           ,'alcohol',
                           'malic_acid',
                           'ash',
                           'alcalinity_of_ash',
                           'magnesium',
                           'total_phenols',
                           'flavanoids',
                           'nonflavanoid_phenols',
                           'proanthocyanins',
                           'color_intensity',
                           'hue',
                           'OD280/OD315',
                           'proline']

        dataset.head()
        self.x = dataset.values[:, 1:]
        self.y = dataset.values[:, 0]

   def _division_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=0)
       
       #Feature Scaling
        self.scaler = StandardScaler()
        self.scaler.fit(self.x_train)

        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

   def _train_model(self, height):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=height, random_state=0)
        model.fit(self.x_train, self.y_train)
        return model        
    
   def plot_tree(self):
        tree.plot_tree(self.model)
        
        
   def train_model(self):
        
        self._division_train_test()
        
        #model = _train_model()
        
        for height in range(1, 21):
            self.model = self._train_model(height)
            self.y_pred = self.model.predict(self.x_test)
            
            print('-----------------------------------------------\n')
            print(f'Altura - {height}\n')
            print("Precisão: "+ str(accuracy_score(self.y_test, self.y_pred)))
           
        self.model = self._train_model(8) 
        


arvoredecisao = ArvoreDecisao('wine.data')     

arvoredecisao.train_model()   

arvoredecisao.plot_tree()
            
                     
    
    




























