# MLB_players
Machine learning algorithms and baseball.

Aiming to implement a k-nearest neighbors algorithm to predict an MLB player's position based on their offensive statistics. The algorithm uses a random sample of 80% of our data, with the remaining 20% used as a test to record the error of the algorithm as a ratio of number of incorrect guesses to size of test sample.

Players.py produces predictions using n neighbours in our model, where n ranges between 1 and 70 to try to optimize error. We then perform our test 50 times in an attempt to reduce the influence of variance when choosing the number of neighbours. This helps us choose the optimal number of neighbors, as well as estimate the error our model is likely to produce.

All data obtained from Baseball Reference, October 24th, 2019.
baseball-reference.com

