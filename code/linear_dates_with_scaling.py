#!/usr/bin/env python

"the benchmark with date features"
"0.31787 on public leaderboard"

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

#

train_file = '../data/train.csv'
test_file = '../data/test.csv'

output_file = 'linear_regression_with_date_features_and_scaling.csv'

#

train = pd.read_csv( train_file, parse_dates = [ 'date' ])
test = pd.read_csv( test_file, parse_dates = [ 'date' ])

train = train.dropna( axis = 0, how = 'any' )

# day of week & month features

train['day_of_week'] = train.date.dt.dayofweek
train['month'] = train.date.dt.month

test['day_of_week'] = test.date.dt.dayofweek
test['month'] = test.date.dt.month

train = pd.get_dummies( train, columns = [ 'day_of_week', 'month' ])
test = pd.get_dummies( test, columns = [ 'day_of_week', 'month' ])

#

x_train = train.drop([ 'Id', 'date', 'region', 'mortality_rate' ], axis = 1 )
y_train = train.mortality_rate.values

x_test = test.drop([ 'Id', 'date', 'region' ], axis = 1 )

#

scaler = MinMaxScaler()
x_train_ = scaler.fit_transform( x_train )
x_test_ = scaler.transform( x_test )

lr = LinearRegression()
lr.fit( x_train_, y_train )

coefs = zip( x_train.columns, lr.coef_ )
for c, i in sorted( coefs, key = lambda _: _[1] ):
	print "{:.1f} {}".format( i, c )

p = lr.predict( x_test_ )

predictions = test[[ 'Id' ]].copy()
predictions[ 'mortality_rate' ] = p

#predictions.to_csv( output_file, index = False )
