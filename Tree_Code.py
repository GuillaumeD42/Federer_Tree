# Libraires Import

import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

# Data Import & Exploration

Federer_raw = pd.read_csv('roger-federer_1998-2016.csv', parse_dates=True, index_col='tourney_dates')


# Data Analysis

Federer_raw['Victory'] = Federer_raw.match_win_loss.map({'L' : 0, 'W' : 100})
Federer_copy = Federer_raw.copy()
Federer_copy = Federer_raw[Federer_raw.opponent_rank != '-']
Federer = Federer_copy.copy()

# Filling in the missing values for these columns with the median of each ones
Federer.player_first_serve_points_won_percentage.fillna(Federer.player_first_serve_points_won_percentage.median(), inplace=True)
Federer.player_first_serve_percentage.fillna(Federer.player_first_serve_percentage.median(), inplace=True)
Federer.games_won.fillna(Federer.games_won.median(), inplace=True)
Federer.match_duration.fillna(Federer.match_duration.median(), inplace=True)
Federer.player_aces.fillna(Federer.player_aces.median(), inplace=True)

# Types

Federer['player_first_serve_percentage'] = Federer['player_first_serve_percentage'].astype(int)
Federer['opponent_rank'] = Federer['opponent_rank'].astype(str).astype(int)
Federer['player_ranking'] = Federer['player_ranking'].astype(int)
Federer['match_duration'] = Federer['match_duration'].astype(int)
Federer['player_aces'] = Federer['player_aces'].astype(int)
Federer['player_first_serve_points_won_percentage'] = Federer['player_first_serve_points_won_percentage'].astype(int)

# Prediction of the mean of the entire dataset
Federer['prediction'] = Federer.Victory.mean()
Federer.head()

# RMSE for those predictions
from sklearn import metrics
import numpy as np
np.sqrt(metrics.mean_squared_error(Federer.Victory, Federer.prediction))

# RMSE for a given split of first serve points won (fspw)
def firstservepc_split(player_first_serve_points_won_percentage):
    lower_firstserve_win = Federer[Federer.player_first_serve_points_won_percentage < player_first_serve_points_won_percentage].Victory.mean()
    higher_firstserve_win = Federer[Federer.player_first_serve_points_won_percentage >= player_first_serve_points_won_percentage].Victory.mean()
    Federer['prediction'] = np.where(Federer.player_first_serve_points_won_percentage < player_first_serve_points_won_percentage, lower_firstserve_win, higher_firstserve_win)
    return np.sqrt(metrics.mean_squared_error(Federer.player_first_serve_points_won_percentage, Federer.prediction))

# calculate RMSE for tree which splits on fspw < 70
print("RMSE :", firstservepc_split(72))
Federer

# check all possible mileage fspw
firstserve_range = range(Federer.player_first_serve_points_won_percentage.min(), Federer.player_first_serve_points_won_percentage.max(), 1)
RMSE = [firstservepc_split(player_first_serve_points_won_percentage) for player_first_serve_points_won_percentage in firstserve_range]

# allow plots to appear in the notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

# plot fswp cutpoint (x-axis) versus RMSE (y-axis)
plt.plot(firstserve_range, RMSE)
plt.xlabel('First Serve Cutpoint')
plt.ylabel('RMSE (lower is better)')


## Building a regression tree in scikit-learn

# define X and y
feature_cols = ['player_ranking', 'player_first_serve_percentage', 'opponent_rank', 'match_duration', 'player_first_serve_points_won_percentage', 'player_aces']
X = Federer[feature_cols]
y = Federer.Victory

# instantiate a DecisionTreeRegressor (with random_state=1)
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg

# use leave-one-out cross-validation (LOOCV) to estimate the RMSE for this model
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


## Tuning a regression tree

# try different values one-by-one
treereg = DecisionTreeRegressor(max_depth=1, random_state=1)
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# list of values to try
max_depth_range = range(1, 10)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

# use LOOCV with each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')

# max_depth=3 was best, so fit a tree using that parameter
treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
treereg.fit(X, y)

# "Gini importance" of each feature: the (normalized) total reduction of error brought by that feature
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})


## Creating a tree diagram

# create a Graphviz file
from sklearn.tree import export_graphviz
export_graphviz(treereg, out_file='Tree_Federer_VictoryTest.dot', feature_names=feature_cols)

# At the command line, run this to convert to PNG:
#dot -Tpng 'Tree_Federer'.dot > Fed.png


## Making predictions for the testing data

# read the testing data
Federer

# Fitted model to make predictions on testing data
X_Federer = Federer[feature_cols]
y_Federer = Federer.Victory
y_pred = treereg.predict(X_Federer)
y_pred


# calculate RMSE
np.sqrt(metrics.mean_squared_error(y_Federer, y_pred))
y_Federer = [60, 70, 80]
y_pred = [0, 0, 0]
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_Federer, y_pred))
