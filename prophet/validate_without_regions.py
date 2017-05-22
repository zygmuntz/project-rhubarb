#!/usr/bin/env python

"validate using prophet, using mean region predictions"

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

# some regions have other date spans than others
d['mean_mortality_rate'] = d.groupby( 'date' ).mortality_rate.transform( 'mean' )
d = d.drop_duplicates( 'date' )

d = d[['date', 'mortality_rate']]
d.columns = ['ds', 'y']

d.y = np.log( d.y )

train = d[:split_at].copy().reset_index( drop = True )
test = d[split_at:].copy().reset_index( drop = True )

prophet = Prophet()
prophet.fit( train )
p = prophet.predict( test )

score = sqrt( MSE( np.exp( test.y ), np.exp( p.yhat )))
print 'RMSE: {:.2%}'.format( score )

prophet.plot( p )
prophet.plot_components( p )
plt.show()
