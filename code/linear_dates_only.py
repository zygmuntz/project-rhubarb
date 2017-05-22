#!/usr/bin/env python

"the benchmark with date features only"
"0.30421 on public leaderboard"

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

#

train_file = '../data/train.csv'
test_file = '../data/test.csv'

output_file = 'linear_regression_with_date_features_only.csv'

log_transform = True

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

day_of_week_cols = [ c for c in train.columns if c.startswith( 'day_of_week_' )]
month_cols = [ c for c in train.columns if c.startswith( 'month_' )]

use_cols = day_of_week_cols + month_cols

x_train = train[ use_cols ]
y_train = train.mortality_rate.values

x_test = test[ use_cols ]

#

lr = LinearRegression()
lr.fit( x_train, y_train )

p = lr.predict( x_test )

predictions = test[[ 'Id' ]].copy()
predictions[ 'mortality_rate' ] = p

predictions.to_csv( output_file, index = False )
