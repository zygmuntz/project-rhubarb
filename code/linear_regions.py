#!/usr/bin/env python

"the benchmark with one-hot region features"
"0.35997 on public leaderboard"

import pandas as pd
from sklearn.linear_model import LinearRegression

#

train_file = '../data/train.csv'
test_file = '../data/test.csv'

train = pd.read_csv( train_file, parse_dates = ['date'])
test = pd.read_csv( test_file )

train = train.dropna( axis = 0, how = 'any' )


# encode region
train = pd.get_dummies( train, columns = [ 'region' ])
test = pd.get_dummies( test, columns = [ 'region' ])


x_train = train.drop([ 'Id', 'date', 'mortality_rate' ], axis = 1 )
y_train = train.mortality_rate.values

x_test = test.drop([ 'Id', 'date' ], axis = 1 )

lr = LinearRegression()
lr.fit( x_train, y_train )

p = lr.predict( x_test )

predictions = test[[ 'Id' ]].copy()
predictions[ 'mortality_rate' ] = p

predictions.to_csv( 'linear_regression_with_regions.csv', index = False )
