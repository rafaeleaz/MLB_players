import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


position_cheat_sheet = {2: 'Catcher', 3: 'First Base', 4: 'Second Base', 5: 'Third Base', 6: 'Shortstop',
                        7: 'Left Field', 8: 'Center Field', 9: 'Right Field'}
player_count = 0
players = pd.DataFrame(data=[], columns=['PA', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB', 'SO', 'HBP',
                                         'SH', 'SF', 'GDP', 'SB', 'CS', 'BA', 'OBP', 'SLG', 'OPS', 'Pos'])

# A dataframe containing the season statistics of all MLB players between 1985 and 2019 with at least 250 plate
# appearances and who started at least 60% of their games in one of the positions listed in the cheat sheet.

for i in range(2, 10):
    new_df = pd.read_csv('MLB_' + str(i) + '.csv')
    new_df = new_df.drop(columns=['Rk', 'Name', 'Pos'])
    length = new_df.shape[0]
    new_df.index = [j for j in range(player_count, player_count + length, 1)]
    new_df['Pos'] = [position_cheat_sheet[i]] * length
    player_count += length
    players = players.append(new_df)

small_errors = []  # A list of minimum errors found in each trial.
small_index = []  # A list of number of neighbours whic yield the minimum error on each trial.
##

for k in range(50):
    #
    error_log = []

    players = players.sample(frac=1)
    players.index = [j for j in range(player_count)]
    # Randomizes the order of players, and resets the index to be in order.

    X = players.drop(columns="Pos")
    Y = players['Pos']
    data_size = X.shape[0]
    training_no = int(data_size * 0.8)
    test_no = data_size - training_no
    X_training = X[:training_no]
    Y_training = Y[:training_no]
    X_test = X[training_no:]
    Y_test = Y[training_no:]

    for j in range(1, 71):
        model = KNeighborsClassifier(n_neighbors=j)
        model.fit(X_training, Y_training)
        Y_predict = model.predict(X_test)
        error = 0
        for i in range(Y_test.size):
            if Y_test[training_no + i] != Y_predict[i]:
                error += 1
        error_log.append(error / Y_test.size)

    small_errors.append(min(error_log))
    small_index.append(error_log.index(min(error_log)))

plt.plot(small_index, small_errors, "bo")
plt.show()
# Plots the optimal number of neighbors found in each trial, against the error of each trial.
