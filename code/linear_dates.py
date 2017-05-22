#!/usr/bin/env python

"the benchmark with date features"
"0.35997 on public leaderboard"

import pandas as pd
from sklearn.linear_model import LinearRegression

#

train_file = '../data/train.csv'
test_file = '../data/test.csv'

output_file = 'linear_regression_with_date_features.csv'

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

lr = LinearRegression()
lr.fit( x_train, y_train )

p = lr.predict( x_test )

predictions = test[[ 'Id' ]].copy()
predictions[ 'mortality_rate' ] = p

predictions.to_csv( output_file, index = False )
