#!/usr/bin/env python

"validate using prophet, predictions for each region separately"

import numpy as np
import pandas as pd

from math import sqrt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as MSE

from matplotlib import pyplot as plt

#

train_file = '../data/train.csv'
split_at = -365

d = pd.read_csv( train_file )
regions = {}
trains = {}
tests = {}
prophets = {}
predictions = {}
scores = {}

for r in sorted( d.region.unique()):
	regions[r] = d[ d.region == r ].copy()
	print r, len( regions[r] )
	
for r, df in regions.items():
	df = df[['date', 'mortality_rate']]
	df.columns = ['ds', 'y']
	df.y = np.log( df.y )
	
	trains[r] = df[:split_at].copy().reset_index( drop = True )
	tests[r] = df[split_at:].copy().reset_index( drop = True )
	prophets[r] = Prophet()
	
	prophets[r].fit( trains[r] )
	predictions[r] = prophets[r].predict( tests[r] )
	scores[r] = sqrt( MSE( np.exp( tests[r].y ), np.exp( predictions[r].yhat )))
	
	print '{} RMSE: {:.2%}'.format( r, scores[r] )
	prophets[r].plot( predictions[r] )
	prophets[r].plot_components( predictions[r] )

for r in sorted( regions ):
	print '{} RMSE: {:.2%}'.format( r, scores[r] )
	prophets[r].plot_components( predictions[r] )
	plt.title( r )
	
print '\nAverage RMSE: {:.2%}'.format( np.mean( scores.values()))
plt.show()

"""
E12000001 RMSE: 25.44%
E12000002 RMSE: 19.24%
E12000003 RMSE: 19.05%
E12000004 RMSE: 21.31%
E12000005 RMSE: 21.67%
E12000006 RMSE: 17.84%
E12000007 RMSE: 11.83%
E12000008 RMSE: 14.66%
E12000009 RMSE: 18.53%

Average RMSE: 18.84%
"""
