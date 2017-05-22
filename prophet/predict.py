#!/usr/bin/env python

"train, predict and save a submission file"
"predictions for each region separately"

import numpy as np
import pandas as pd

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

#

train_file = '../data/train.csv'
test_file = '../data/test.csv'
output_file = '../prophet_predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

train_regions = {}
test_regions = {}
prophets = {}
predictions = {}
scores = {}

train = train[[ 'region', 'date', 'mortality_rate' ]]
train.columns = [ 'region', 'ds', 'y' ]
#train.y = np.log( train.y )

test = test[[ 'Id', 'region', 'date' ]]
test.columns = [ 'Id', 'region', 'ds' ]

for r in sorted( train.region.unique()):
	train_regions[r] = train[ train.region == r ].copy()
	test_regions[r] = test[ test.region == r ].reset_index( drop = True )
	print r, len( train_regions[r] ), len( test_regions[r] )
		
for r in train_regions:
	prophets[r] = Prophet()
	prophets[r].fit( train_regions[r] )
	predictions[r] = prophets[r].predict( test_regions[r] )
	predictions[r]['mortality_rate'] = predictions[r].yhat

for r in train_regions:
	print predictions[r].head()
	prophets[r].plot( predictions[r] )
	#prophets[r].plot_components( predictions[r] )
plt.show()

submissions = []
for r in predictions:
	tmp = predictions[r][['Id', 'mortality_rate']]
	submissions.append( tmp )
	
s = pd.concat( submissions )
s = s.sort_values( 'Id' )
s.to_csv( output_file, index = None )
