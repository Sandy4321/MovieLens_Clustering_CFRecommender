import numpy as np
import scipy

X = np.array(np.genfromtxt('u.data', delimiter = '	'))
users = X[:, 0]
movies = X[:, 1]
ratings = X[:, 2]
uniqueUsers = set(list(users))
uniqueMovies = set(list(movies))

sparseMatrix = np.empty((0,len(uniqueMovies)), int)
movieRow = np.zeros(len(uniqueMovies))


for user in uniqueUsers:
	matches = np.where(users == user)[0]
	#movies are 1 to n, we'll shift the index to 0 to n - 1
	indices = movies[matches] - 1
	userRatings = 1.0 * ratings[matches]
	thisRow = movieRow
	thisRow.put(indices.astype(int), userRatings)
	sparseMatrix = np.append(sparseMatrix, [thisRow], axis=0)

np.savetxt("sparseMovies.csv", sparseMatrix, delimiter=",")

def getTopNSimilarItems(X, entity, target = "users", distanceMatrix = None):
	if target != "users":
		X = X.T
	
	if distanceMatrix == None:
		#dist = spatial.distance.pdist(X, metric='euclidean')
		#distanceMatrix = spatial.distance.squareform(dist)
		#np.savetxt("distanceMatrixEuclid.csv", distanceMatrix, delimiter=",")
		#dist = spatial.distance.pdist(X, metric='correlation')
		#distanceMatrix = spatial.distance.squareform(dist)
		#np.savetxt("distanceMatrixCorrelation.csv", distanceMatrix, delimiter=",")
		dist = spatial.distance.pdist(X, metric= lambda x, y: stats.pearsonr(x, y)[0])
		distanceMatrix = spatial.distance.squareform(dist)
		np.savetxt("distanceMatrixPearson.csv", distanceMatrix, delimiter=",")
		print distanceMatrix
		
		#otherUserRatings = X[np.arange(X.shape[0]) != userIndex, :]
	

		#weightedRatings = (otherUserRatings.T * similarities).T
		#Remove the actual user's ratings
		#weightedRatings = weightedRatings[np.arange(X.shape[0]) != userIndex, :]
