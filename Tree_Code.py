# In[2]:

import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
```

# In[3]:

Federer_raw = pd.read_csv('roger-federer_1998-2016.csv', parse_dates=True, index_col='tourney_dates')


# In[4]:

Federer_raw['Victory'] = Federer_raw.match_win_loss.map({'L' : 0, 'W' : 100})


# In[5]:

Federer_copy = Federer_raw.copy()


# In[6]:

Federer_copy = Federer_raw[Federer_raw.opponent_rank != '-']


# In[7]:

Federer = Federer_copy.copy()


# In[8]:

# fill in the missing values for these columns with the median of each ones
Federer.player_first_serve_points_won_percentage.fillna(Federer.player_first_serve_points_won_percentage.median(), inplace=True)
Federer.player_first_serve_percentage.fillna(Federer.player_first_serve_percentage.median(), inplace=True)
Federer.games_won.fillna(Federer.games_won.median(), inplace=True)
Federer.match_duration.fillna(Federer.match_duration.median(), inplace=True)
Federer.player_aces.fillna(Federer.player_aces.median(), inplace=True)


# In[9]:

Federer['player_first_serve_percentage'] = Federer['player_first_serve_percentage'].astype(int)
Federer['opponent_rank'] = Federer['opponent_rank'].astype(str).astype(int)
Federer['player_ranking'] = Federer['player_ranking'].astype(int)
Federer['match_duration'] = Federer['match_duration'].astype(int)
Federer['player_aces'] = Federer['player_aces'].astype(int)
Federer['player_first_serve_points_won_percentage'] = Federer['player_first_serve_points_won_percentage'].astype(int)


# In[10]:

# before splitting anything, just predict the mean of the entire dataset
Federer['prediction'] = Federer.Victory.mean()
Federer.head()


# In[11]:

# calculate RMSE for those predictions
from sklearn import metrics
import numpy as np
np.sqrt(metrics.mean_squared_error(Federer.Victory, Federer.prediction))


# In[12]:

# define a function that calculates the RMSE for a given split of first serve points won (fspw)
def firstservepc_split(player_first_serve_points_won_percentage):
    lower_firstserve_win = Federer[Federer.player_first_serve_points_won_percentage < player_first_serve_points_won_percentage].Victory.mean()
    higher_firstserve_win = Federer[Federer.player_first_serve_points_won_percentage >= player_first_serve_points_won_percentage].Victory.mean()
    Federer['prediction'] = np.where(Federer.player_first_serve_points_won_percentage < player_first_serve_points_won_percentage, lower_firstserve_win, higher_firstserve_win)
    return np.sqrt(metrics.mean_squared_error(Federer.player_first_serve_points_won_percentage, Federer.prediction))


# In[13]:

# calculate RMSE for tree which splits on fspw < 70
print("RMSE :", firstservepc_split(72))
Federer


# In[14]:

# check all possible mileage fspw
firstserve_range = range(Federer.player_first_serve_points_won_percentage.min(), Federer.player_first_serve_points_won_percentage.max(), 1)
RMSE = [firstservepc_split(player_first_serve_points_won_percentage) for player_first_serve_points_won_percentage in firstserve_range]


# In[15]:

# allow plots to appear in the notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14


# In[16]:

# plot fswp cutpoint (x-axis) versus RMSE (y-axis)
plt.plot(firstserve_range, RMSE)
plt.xlabel('First Serve Cutpoint')
plt.ylabel('RMSE (lower is better)')


# ## Building a regression tree in scikit-learn

# In[17]:

# define X and y
feature_cols = ['player_ranking', 'player_first_serve_percentage', 'opponent_rank', 'match_duration', 'player_first_serve_points_won_percentage', 'player_aces']
X = Federer[feature_cols]
y = Federer.Victory


# In[18]:

# instantiate a DecisionTreeRegressor (with random_state=1)
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg


# In[19]:

# use leave-one-out cross-validation (LOOCV) to estimate the RMSE for this model
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


# ## Tuning a regression tree

# In[20]:

# try different values one-by-one
treereg = DecisionTreeRegressor(max_depth=1, random_state=1)
scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


# Or, we could write a loop to try a range of values:

# In[21]:

# list of values to try
max_depth_range = range(1, 10)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

# use LOOCV with each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# In[22]:

# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')


# In[23]:

# max_depth=3 was best, so fit a tree using that parameter
treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
treereg.fit(X, y)


# In[24]:

# "Gini importance" of each feature: the (normalized) total reduction of error brought by that feature
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})


# ## Creating a tree diagram

# In[25]:

# create a Graphviz file
from sklearn.tree import export_graphviz
export_graphviz(treereg, out_file='Tree_Federer_VictoryTest.dot', feature_names=feature_cols)


# In[26]:

# At the command line, run this to convert to PNG:
#dot -Tpng 'Tree_Federer'.dot > Fed.png


# ## Making predictions for the testing data

# In[27]:

# read the testing data
Federer


# Question: Using the tree diagram above, what predictions will the model make for each observation?

# In[28]:

# use fitted model to make predictions on testing data
X_Federer = Federer[feature_cols]
y_Federer = Federer.Victory
y_pred = treereg.predict(X_Federer)
y_pred


# In[29]:

# calculate RMSE
np.sqrt(metrics.mean_squared_error(y_Federer, y_pred))


# In[30]:

# calculate RMSE for your own tree!
y_Federer = [60, 70, 80]
y_pred = [0, 0, 0]
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_Federer, y_pred))