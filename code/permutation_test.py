#!/usr/bin/env python

"perform a permutation test"

import pandas as pd

from math import sqrt
from sklearn.model_selection import permutation_test_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
from matplotlib import pyplot as plt

#

train_file = '../data/train.csv'

n_estimators = 100

cv = 2					# 2-fold validation, for speed
n_permutations = 100 

#

d = pd.read_csv( train_file, parse_dates = [ 'date' ])
d = d.dropna( axis = 0, how = 'any' )

d = pd.get_dummies( d, columns = [ 'region' ])

# binary indicators of train/test for permutation_test_score
train_i = d.date.dt.year < 2012
test_i = d.date.dt.year == 2012

x = d.drop([ 'Id', 'date', 'mortality_rate' ], axis = 1 ).values
y = d.mortality_rate.values

#

clf = GBR( n_estimators = n_estimators )

"""
score, permutation_scores, pvalue = permutation_test_score(
    clf, x_train, y_train, scoring = "neg_mean_squared_error", 
    cv = cv, n_permutations = n_permutations )
"""

neg_mse, permutation_scores, pvalue = permutation_test_score(
    clf, x, y, scoring = "neg_mean_squared_error", 
    cv = [[ train_i, test_i ]], n_permutations = n_permutations )    

rmse = sqrt( abs( neg_mse ))
print "RMSE: {:.4f}, p value : {:.2%}".format( rmse, pvalue )

#

plt.hist( permutation_scores, 20, label = 'Permutation scores' )
ylim = plt.ylim()

plt.plot( 2 * [neg_mse], ylim, '--g', linewidth=3, label='MSE (p value: {:.2%})'.format( pvalue ))

plt.ylim( ylim )
plt.legend()
plt.xlabel( 'Score' )
plt.show()
