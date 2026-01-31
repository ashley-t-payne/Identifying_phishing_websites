#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:12:52 2025

@author: ashleypayne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations


phishingDF = pd.read_csv("phishing_website_training_data.csv") 



'''
for col in phishingDF:
    print(col)
    
#VISUALIZATIONS
X = phishingDF.drop('Result', axis=1)
X = phishingDF.drop('id', axis = 1)
y = phishingDF['Result']

phish = phishingDF[phishingDF["Result"] == 1]  # Class 1 = Divorced
not_phish = phishingDF[phishingDF["Result"] == -1]  # Class 0 = Not Divorced


plt.figure(figsize=(20, 30))  # Adjust figure size

attributes = phishingDF.drop(columns=["Result"]) # Get all feature columns

# Create boxplots for each attribute
for i, attribute in enumerate(attributes, 1):
    data_phish = phish[attribute]
    data_not_phish = not_phish[attribute]

    plt.subplot(8, 4, i)  # Adjust grid size based on number of attributes
    plt.boxplot([data_not_phish, data_phish], tick_labels=["Not A Phish", "Phish"], patch_artist=True)
    plt.title(attribute)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(20, 20))  # 20x20 inches figure

axs = axs.flatten()  # Flatten 2D array of axes to 1D for easy looping
#Create Pie Charts
for i, attribute in enumerate(attributes, 1):
    # Count how many phishing and not-phishing samples for each attribute value
    data_phish = phish[attribute].value_counts()
    data_not_phish = not_phish[attribute].value_counts()
    

    # Sum total counts (you can adjust depending on how you want to handle missing values)
    values = data_not_phish.values/data_not_phish.sum()
    z = data_phish.values/data_phish.sum()
    #print(data_not_phish)
    
    plt.figure()
    
    plt.subplot(1, 2, 1)  # 8 rows, 4 columns grid
    plt.pie(values, labels=data_not_phish.keys(), autopct='%1.1f%%', startangle=140)
    plt.subplot(1, 2, 2)  # 8 rows, 4 columns grid

    plt.pie(z, labels=data_phish.keys(), autopct='%1.1f%%', startangle=140)
    plt.title(attribute)
    plt.show()
    
    

plt.tight_layout()
plt.show()
plt.show()


PIE CHART CONCLUSIONS:
    web_trafic
    SFH
    URL_of_Anchor
    Request_URL*
    Domain_registration_length
    SSLfinal_State*
    having_Sub_Domain
    Prefix_Suffix*
    URL_of_Anchor



for attribute in attributes:
    correlation = np.corrcoef(phishingDF[attribute], y)[0, 1]
    #if(correlation > 0.63):
    print(attribute, ': ' , correlation)

#two of the highest correlations were SSLfinal_State, and URL_of_Anchor

'''
X = phishingDF[['web_traffic', 'SFH', 'URL_of_Anchor', 'Request_URL', 'Domain_registeration_length',
             'SSLfinal_State', 'having_Sub_Domain', 'Prefix_Suffix', 'URL_of_Anchor']]
y = phishingDF['Result']

model = DecisionTreeClassifier()

for comb in combinations(X, 1):
    sets = phishingDF[list(comb)]
    results = cross_validate(model, sets, y, cv=5, scoring = ['accuracy', 'precision_macro', 'recall_macro'])
    
    acc = np.mean(results['test_accuracy'])
    pre = np.mean(results['test_precision_macro'])
    rec = np.mean(results['test_recall_macro'])
    print("acc=%.2f, pre=%.2f, rec=%.2f" % (acc, pre, rec))

print()
    
#test

def print_metrics(results):
    acc = np.mean(results['test_accuracy'])
    pre = np.mean(results['test_precision_macro'])
    rec = np.mean(results['test_recall_macro'])
    print("acc=%.3f, pre=%.3f, rec=%.3f" % (acc, pre, rec))
    
def print_tree(results):
    for tree in results['estimator']:
        plt.figure(figsize=(20, 30))
        plot_tree(tree)
        plt.show()
        
model = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.005)
results = cross_validate(model, X, y, cv=5, return_estimator=True, scoring = ['accuracy', 'precision_macro', 'recall_macro'])
print_metrics(results)
print_tree(results)

print()



'''
#predict
test_df = testdf[['Atr9', 'Atr11', 'Atr17', 'Atr18', 'Atr19', 'Atr40']]
model.fit(X, y) #trains model
y_pred_train = model.predict(test_df)
test_df['class_prediction'] = y_pred_train
test_df.to_csv('Payne.csv', index=False)
'''