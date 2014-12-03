import numpy as np
import scipy
from sklearn.cluster import DBSCAN
from sklearn import mixture
from statsmodels.distributions.empirical_distribution import ECDF

def rankAssociationClustering(X, similarity = "strict", epsilon = 0.0001,  min_smpls = 2, weights = None):
	rankMethod = "min"
	if weights is not None:
		X = weights * X

	rankedX = np.apply_along_axis(lambda x: scipy.stats.rankdata(-1.0 * x, method = "min"), 1, X)
	uniqueX = np.vstack({tuple(row) for row in rankedX})
	#print uniqueX
	if similarity == "strict":
		finalClusters = []
		for unique_rank in uniqueX:
			matches = np.where((rankedX == unique_rank).all(axis=1))[0]
			finalClusters.append(list(matches))

	else:
		if similarity == "rho":
			db = DBSCAN(eps=epsilon, min_samples = min_smpls,algorithm="brute", metric = lambda x, y: (-1.0*(scipy.stats.spearmanr(x, y)[0] - 1))).fit(uniqueX)
		else:
			db = DBSCAN(eps=epsilon, min_samples = min_smpls, algorithm="brute", metric = lambda x, y: (-1.0*(scipy.stats.kendalltau(x, y)[0] - 1))).fit(uniqueX)
		labels = db.labels_

		clusters = []
		for label in set(labels):
			indexList = label == labels
			indices = list(np.where(indexList)[0])
			if label == -1:
				for id in indices:
					clusters.append([id])
			else:
				clusters.append(indices)

		finalClusters = []
		for cluster in clusters:
			finalCluster = []
			for unique_rank in cluster:
				matches = np.where((rankedX == uniqueX[unique_rank]).all(axis=1))[0]
				finalCluster.extend(matches)
			finalClusters.append(finalCluster)

	labels_, n_clusters = labelClusters(finalClusters)
	return finalClusters, labels_, n_clusters, rankedX

def labelClusters(clusters):
	holderDict = {}
	label = 0
	for cluster in clusters:
		if len(cluster) > 1:
			newDict = dict((element, label) for element in cluster)
			holderDict = dict(holderDict, **newDict)	
			label = label + 1
		else:
			holderDict[cluster[0]] = -1
	return np.array(holderDict.values()), (label + 1)

def QuanfityEccentricity(X, k, thresholdDrop, ecdfFunc = None):		
	candidates = np.ones(X.shape[1])
	current = np.zeros(X.shape[1])
	initialMeans, initialStds, initialBic = np.apply_along_axis(getGMMFunc, 0, X, 1)
	means = initialMeans
	stds = initialStds
	
	for i in range(k - 1):
		n = i + 2
	
	finalX = [np.zeros(X.shape[0])]
	for i in range(X.shape[1]):
		j = 0
		for columnMeans in means[i]:
			currentMean = columnMeans[0]
			currentStd = stds[i][j][0]
			column = X[:, i]
			#z_score = (X[:, i] - currentMean)/currentStd
			#eccentricities = scipy.stats.norm.cdf(z_score)
			#print eccentricities
			#column = z_score
			if ecdfFunc == None:
				eccentricities = ECDF(column)(column)
			else:
				eccentricities = ecdfFunc(column)
			
			#eccentricities = scipy.stats.norm.cdf(z_score)
			eccentricities = (eccentricities-min(eccentricities))/(max(eccentricities)-min(eccentricities))
			#print eccentricities
			#print "----------"
			#eccentricities = ECDF(z_score)(z_score)
			finalX = np.append(finalX, [eccentricities], axis = 0)
			j = j + 1
			
	X = finalX.T[:, 1:]
	return X

def getTopKFeatures(X, k = 2):
	#_, _, bicVector1 = np.apply_along_axis(getGMMFunc, 0, X, 1)
	#_, _, bicVector2 = np.apply_along_axis(getGMMFunc, 0, X, 2)
	#bicVectorDropRatio = (bicVector2 - bicVector1)/bicVector1
	basis = -scipy.stats.variation(X, axis = 0)
	if k < X.shape[1]:
		ranked = scipy.stats.rankdata(basis)
		filteredRanks = np.where(ranked <= k)[0]
		return X[:, filteredRanks]
	else:
		return X

def getContexualWeights(X):
	cv = scipy.stats.variation(X, axis = 0)
	return cv/np.sum(cv)

