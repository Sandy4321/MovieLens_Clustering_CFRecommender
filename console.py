import numpy as np
import scipy
from scipy import spatial
from scipy import stats
import pandas as pd
import finalAlgo as fa

X = np.array(np.genfromtxt('datasets/sparseMovies.csv', delimiter = ','))
movieInfo = similarityMatrix = pd.read_csv("datasets/u.item", header=None, delimiter='|')
similarityMatrix = 1/(1 + np.array(np.genfromtxt('datasets/distanceMatrixCosine.csv', delimiter = ',')))





def clusterUsers(X, limitFeatures = 100, weight = True):
	k = limitFeatures
	X = fa.getTopKFeatures(X, k)
	
	if weight == True:
		thisWeights = fa.getContexualWeights(X)
	#thisWeights = None
	return fa.rankAssociationClustering(X, similarity = "strict", epsilon = 0.1,  min_smpls = 2, weights = thisWeights )
		
def recommendMovies(userIndex, X, similarityMatrix, movieInfo=None, n = 10):
	userRatings = X[userIndex, :]
	candidateMovies = np.where(userRatings == 0)[0]
	similarities = similarityMatrix[userIndex, :]
	print similarities


	otherUserRatings = X[:, candidateMovies]
	predictedRatings = []
	for movieRatings in otherUserRatings.T:
		validIndices = movieRatings > 0
		weights = similarities[validIndices]
		ratings = movieRatings[validIndices]
		weightedRatings = ratings * weights
		totalWeightedRatings = np.sum(weightedRatings)
		totalWeights = np.sum(similarities)
		
		predictedRating = totalWeightedRatings/totalWeights
		predictedRatings.append(predictedRating)

	reverseSorted = sorted(enumerate(predictedRatings), key=lambda x:x[1], reverse = True)
	topMovies = [i[0] for i in reverseSorted]
	topPredictedRating = [i[1] for i in reverseSorted]

	actualTopMovies = candidateMovies[topMovies]
	return actualTopMovies[:n], topPredictedRating[:n], movieInfo.iloc[actualTopMovies, 1]

#indices, ratings, movieNames = recommendMovies(0, X, similarityMatrix, movieInfo, n = 10)
clusters, labels_, n_clusters, rankedX = clusterUsers(X.T, limitFeatures = 10, weight = True)
for c in clusters:
	print movieInfo.iloc[c, 1]
	print "--------------------"