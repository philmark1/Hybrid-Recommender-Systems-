# -*- coding: utf-8 -*-
"""
# 1509 Applications of Data Science
## Movie Recommendation System Using Item Based Collaborative Filtering
Publications of the following authors were used: Poonam Sharma, Lokesh Yadav

By Philipp Markopuloss (h12030674)

### Process
1. Ratings Data Preprocessing (user-item-rating-matrix)
2. Keywords Data Preprocessing (item-keywords-binary-matrix)
3. Content-based filtering algorithm
6. Item-based collaborative filtering algorithm
7. Hybrid RecSys algorithm
8. Testing
10. Evaluation
"""

import pandas as pd
import numpy as np
import json
import ast
from pandas.api.types import CategoricalDtype
import sys
from IPython.display import clear_output
import time
import csv

# These Libraries carry the whole implementation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

"""## Setting a limit on the datasize
this is necessary to compute fast results and display matrices with Google Collabs RAM limitations

the algorithms remain the same, its just faster
"""

# set a limit on how many users and movies you want to include
# at these limits(200,200) the whole code runs within few minutes
# the whole dataset runs for 1.5 days
usersLimit = 200
movieLimit = 200

# sparse pivot table creator
def sparsePivotTable(df, rowCol, colCol, valCol):
  # create factor arrays for the rows and columns
  rowCol_cat = CategoricalDtype(sorted(df[rowCol].unique()), ordered=True)
  colCol_cat = CategoricalDtype(sorted(df[colCol].unique()), ordered=True)

  # create the rows and columns from the respective factors
  row = df[rowCol].astype(rowCol_cat).cat.codes
  col = df[colCol].astype(colCol_cat).cat.codes
  
  # create the sparse matrix
  # 1: pass the values for the cells array
  # 2: pass the indices (row) and the names (col)
  # 3: pass the dimensions (length of row and length of col)
  sparse_matrix = csr_matrix((df[valCol], (row, col)), shape=(rowCol_cat.categories.size, colCol_cat.categories.size))
  # turn it into a sparse Pandas Dataframe
  dfs = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
  #return dfs
  return sparse_matrix

# turning a sprase pivot table into a pd.dataframe (optional)
def getSparseDf(sparse_matrix):
  return pd.DataFrame.sparse.from_spmatrix(sparse_matrix)

"""### 1. Ratings Data Preprocessing
create the user-item-ratings-matrix
"""

# read the data from the csv
ratings = pd.read_csv("ratings_small.csv")
ratings = ratings[ratings['userId']<usersLimit]
ratings = ratings[ratings['movieId']<movieLimit]
# call the Sparse Pivot Table Function
ratings = sparsePivotTable(ratings, "userId", "movieId", "rating").astype(int)
display(ratings.todense())
print(ratings.shape)
print("Filesize of the matrix:", sys.getsizeof(ratings)) # this is significantly smaller than creating it entirely with Pandas Pivot Tables

if movieLimit < 1000:
  r = getSparseDf(ratings)
r

"""### 2. Keywords Data Preprocessing
get the item-keyword-binary-matrix
"""

# Load the Keywords Dataset
# 1st col: movieId
# 2nd col: array of dictionaries (2 key-item pairs: keyword id, keyword name) as string
movie_keywords_list = pd.read_csv("keywords.csv")
movie_keywords_list = movie_keywords_list[movie_keywords_list['id']<movieLimit]
# turn the strings into python lists
# cannot use JSON functions, as JSON format rules are not met (single quotes instead of double quotes)
# literal_eval takes the literal meaning of the string as Python code, which results in valid lists
movie_keywords_list.keywords = movie_keywords_list.keywords.apply(lambda x: ast.literal_eval(x))
#display(movie_keywords_list)

# rather than having one row per movie with an array of keyword ids
# … we want a df with one row per keyword per movie
movie_keywords_longer = {}
j = 0
for i, row in movie_keywords_list.iterrows():
  for keyword in row.keywords:
    movie_keywords_longer[j] = {'movieId': row.id, 'keyword': keyword['id']}
    j += 1
# rows and columns are swapped, transpose it
movie_keywords_longer = pd.DataFrame(movie_keywords_longer).transpose()
# remove duplicates
movie_keywords_longer = movie_keywords_longer.drop_duplicates()
# create a column with the value 1; this will be the value for every match in the pivot table
movie_keywords_longer["match"] = 1
#display(movie_keywords_longer)

# call the Sparse Pivot Table Function
movie_keywords = sparsePivotTable(movie_keywords_longer, "movieId", "keyword", "match")
display(movie_keywords)
display(movie_keywords.todense())
print("Filesize of the matrix:", sys.getsizeof(movie_keywords))

if movieLimit < 1000:
  mk = getSparseDf(movie_keywords)
mk

# make a df with all keywords (optional)
# use list comprehension to gather all keyword dictionaries
keywords_list = [keyword for row in movie_keywords_list.keywords for keyword in row]
# make a Pandas DF from it
keywords_df = pd.DataFrame(keywords_list)
keywords_df = keywords_df.drop_duplicates()
keywords_df = keywords_df.set_index('id')
keywords_df = keywords_df.sort_index()
keywords_df

"""### 3. Content-based filtering algorithm
this algorithm creates a user-keyword-profile based on his likes and compares it to all films, the best matches are returned as recommendations
"""

# creating user-profiles frmo their likes
def getUserProfile(ratings, userId, movieKeywordsMatrix):
  """
  Input: 
  ratings: 1-5 int user-ratings-matrix (sparse, 0 if unrated)
  userId: int of user you want the profile of
  movieKeywordsMatrix: binary movie-keyword-matrix (sparse)
  Output: 
  userProfile: binary array for user liking a keyword
  """
  userRatings = ratings[userId]
  # assume that a rating of 3 or higher means that a film is liked
  likingThreshold = 3
  
  # turn the ordinal ratings into binary to indicate liking a movie
  # this requires a dense array, which is not problematic, as we are only looking at one user profile
  movieLikes = userRatings
  movieLikes = np.array(userRatings.todense())
  movieLikes = np.where(movieLikes >= likingThreshold)[1]
  
  # get a list of ids of the movies the user liked
  movieLikeIds = [likeId for likeId, like in enumerate(movieLikes) if like]    
  # select the liked ids from the movies*keywords matrix
  likedMoviesKeywords = movieKeywordsMatrix[movieLikeIds]
  # now we have a matrix with one row for each liked movie and a binary match for each keyword
  # sum the matches vertically 
  userProfile = likedMoviesKeywords.sum(axis=0)

  # now reduce the numbers to be either 1 for liking a keyword or else 0
  userProfile = userProfile
  userProfile = (userProfile>0).astype(int)

  print("User Profile created")
  return userProfile

# Example
getUserProfile(ratings, 0, movie_keywords)

# function for comparing a profile with all movies' profiles to get similarities
def getKeywordProfileSim(userProfile, movieKeywordsMatrix):
  #print("Calculating similarities")
  # loop through all movies and calculate cosine similarity to userProfile
  #sim = [(id, cosine_similarity(userProfile, filmActive)[0][0]) for id, filmActive in enumerate(movieKeywordsMatrix)]
  movie_keywords = movieKeywordsMatrix
  sim = cosine_similarity(userProfile, movie_keywords)
  return sim

# function for comparing a user-profile with all movie profiles to get similarities
def getUserMovieSimilarity(userProfile, movieKeywordsMatrix):
  # reshape the user profile to be a 1d matrix (necessary to get cosine similarity)
  userProfile = np.array(userProfile)
  sim = getKeywordProfileSim(userProfile, movieKeywordsMatrix)
  return sim

# Example
user = getUserProfile(ratings, 69, movie_keywords)
getUserMovieSimilarity(user, movie_keywords)

"""$predictedRatingCB(u_x, i_y)= Σ^i_0sim(movie\_keyword[y,], movie\_keyword[,i])*ratings[x,y]$"""

# perform content based filtering to get n recommendations for a user
def contentBasedFiltering(ratingsArr, userId, movieKeywords, n=10):
  """
  returns top n movies to recommend and their similarity to the user profile
  """
  # get a userProfile(binary array. 1 indicates the user likes the keyword)
  userProfile = getUserProfile(ratingsArr, userId, movieKeywords)
  # get the similarities between each movie and the user profile
  simRowCos = getKeywordProfileSim(userProfile, movieKeywords)
  # get the ids of the top similarities
  topSimilarMovieIds = np.argsort(simRowCos)[0][::-1][:n]
  topSims = simRowCos[0][topSimilarMovieIds]
  return [(id, simRowCos[0][id]) for id in topSimilarMovieIds]
  
# Example
userId = 1
contentBasedFilteringRec = contentBasedFiltering(ratings, userId, movie_keywords)
print("Content-based recommendations for user", userId)
contentBasedFilteringRec

"""### 4. Item-based collaborative filtering algorithm

"""

# getting item-based similarities between all items 
# transpose the matrix before to allow for column-wise calcuations
rating_movie_similarities = cosine_similarity(ratings.transpose())

# function for getting the ids of the movies most similar to the active one
def getMostSimilarMovie(movieId, similarities):
  # get the similarities of that particular movie
  similarities = similarities[movieId]
  similarities = np.delete(similarities, movieId)
  mostSimilarId = np.argmax(similarities)
  return (mostSimilarId, similarities[mostSimilarId])

# getting the best n recommendations for a user

def getIBCF(userId, ratingsMatrix, similaritiesMatrix = "nothing", n=10):
  # this should not be done everytime the function is called
  # but in case someone doesnt compute the similarities beforehand, we do it here
  if similaritiesMatrix == "nothing":
    # getting similarities between ratings 
    # transpose the matrix before to allow for column-wise calcuations
    similaritiesMatrix = cosine_similarity(ratingsMatrix.transpose())
  # get the sparse ratings for the user
  userRatings = ratings[userId,:].todense()
  # get the indices of the rated items
  userRatingIds = userRatings.nonzero()
  # make get a matrix with the indices and the corresponding ratings
  userRating = np.array([
      userRatingIds[1],
      np.array(userRatings[userRatingIds])[0]
      ])
  # create an argsort array; this returns the indices to grab to get the array in order
  sortBy = np.argsort(userRating[1])
  sortBy = np.flip(sortBy)
  # recreate the 2d-array, but use the sorted indices
  userRating =userRating[0][sortBy]
  # we need integer indices for indexing later …
  userRating = userRating.astype(int)
  # get the most similar movies for the top n rated movies
  itemBasedCollabFilteringRec = [getMostSimilarMovie(movieId, rating_movie_similarities) for movieId in userRating[:n]]
  return itemBasedCollabFilteringRec

# Example
userId = 1
print("Item-based recommendations for user", userId)
getIBCF(userId, ratings, rating_movie_similarities)

"""### Predicting movie ratings using IBCF

$predictedRatingIBCF(u_x, i_y)= Σ^i_0sim(ratings[,y], ratings[,i])*ratings[x,y]$
"""

# predicting a movieRating for a user for a certain movie

def predictMovieRatingIBCF(userId, movieId, ratingsMatrix, similaritiesMatrix, n=10):
  # get the similarities of the movies compared to the current movie
  sim = similaritiesMatrix[movieId]
  # get the ids of the n most similar movies
  topSimIds = np.where(np.argsort(sim)<n)
  # get the similarities of the top similar movies
  topSim = sim[topSimIds]
  # normalize their similarities to get the movies ratings weights
  topSimNorm = topSim / sum(topSim)
  # get the ratings of the top n similar movies that the rated
  topSimRatings = ratings[userId].todense()[0,[topSimIds]]
  topSimRatings = np.array(topSimRatings)[0][0]
  # mulitply their ratings with their normalized similarities
  weightedRatings = topSimNorm * topSimRatings
  # sum it up and thats you predicted rating for the movie
  predicTedRating = sum(weightedRatings)
  return round(predicTedRating)

# Example for user 10 and movie 10
predictMovieRatingIBCF(10, 10, ratings, rating_movie_similarities)

"""### 5. Hybrid RecSys Algorithm
Content-Based -> fill sparse ratings matrix with pseudo-ratings

Item-Based Collaborative Filtering -> replace pseudo-ratings with proper rating predictions

"""

# function for creating a pseudo ratings matrix

def getUserPseudoRatings(userId, ratingsMatrix):
  start_time = time.time()
  # select the user's row
  userRow = ratingsMatrix.todense()[userId]
  userRow = np.array(userRow)[0]
  # get the ids of the rated and unrated movies
  ratedIds = userRow != 0
  unratedIds = userRow == 0
  # for each film he hasnt rated
  for unratedFilmId,_ in enumerate(userRow[unratedIds]):
    try:
      unratedFilmProfile = movie_keywords[unratedFilmId]
      # get similarities to all films he rated
      sim = getKeywordProfileSim(unratedFilmProfile, movie_keywords[ratedIds])
      sim = sim[0]
      # normalize the similarities to add up to 1
      sim = sim / sum(sim)
      # multiply each normalized similarity with the rating of the movie you compared the unrated movie to
      weightedRatings = userRow[ratedIds] * sim
      # sum up the weighted ratings and round the result
      pseudoRating = sum(weightedRatings)
      # if the sum is np.nan, you cannot round it, just put it to 0
      if np.isnan(pseudoRating):
        pseudoRating = 0
      pseudoRating = round(pseudoRating)
      # save the pseudoRating to the userRow 
      userRow[userId] = pseudoRating
    except Exception:
      continue
  clear_output()
  print(f"User {userId} done –", time.time() - start_time, "seconds taken")
  #print(userRow)
  ratingsMatrix[0] = userRow
  return ratingsMatrix

# create a copy of the user-item-ratings-matrix to fill in th pseudo ratings
pseudoRatings = ratings.copy()

# loop through the rows of the pseudo-ratings-matrix 
# this way each row is replaced with the content-based predicted ratings 
for i, row in enumerate(pseudoRatings):
  pseudoRatings = getUserPseudoRatings(i, pseudoRatings)

print(pseudoRatings.todense())
print("Filesize of the matrix:", sys.getsizeof(pseudoRatings))

if movieLimit < 1000:
  p = getSparseDf(pseudoRatings)
p

"""### 6. Testing
to test the hybrid algorithm, we predict some already known ratings and get a confusion matrix

now that we have a full pseudo-ratings matrix, we can perform more informed collaborative filtering predictions
"""

# function for making rating predictions of a user and a movie

def getHybridRecommendation(userId, movieId, ratingsMatrix, pseudoRatingsMatrix, rating_movie_similarities):
  if ratingsMatrix[userId, movieId]:
    print("Movie already rated")
    return ratingsMatrix[userId, movieId]
  else:
    predictedRating = predictMovieRatingIBCF(userId, movieId, pseudoRatingsMatrix, rating_movie_similarities)
    return predictedRating

# Example
userId = 1
movieId = 9
rec = getHybridRecommendation(userId, movieId, ratings, pseudoRatings, rating_movie_similarities)
print("Hybrid Recommendation")
print(f"Rating for user {userId} and movie {movieId} predicted:", rec)

# create a copy of the ratings and predict a rating for every unrated movie
ratings_test = ratings.copy()

for i,row in enumerate(ratings_test):
  if i % 500 == 0: print(f"{i} tested")
  # make the row an accessible array
  row = np.array(row.todense())[0]
  # get the ids where the ratings are known
  ratedIds = np.where(row != 0)[0]
  # overwrite the ratings to be 0
  ratings_test[ratedIds] = 0
  # now predict the rating for every previously known rating
  for j in ratedIds:
    ratings_test[i, j] = predictMovieRatingIBCF(i, j, pseudoRatings, rating_movie_similarities)

clear_output()

print(ratings_test.todense())
print(sys.getsizeof(ratings_test))
print(ratings_test.shape)
print(ratings.shape)

if movieLimit < 1000:
  t = getSparseDf(ratings_test)
t

"""### 7. Evaluation
to test the hybrid algorithm, we predict some already known ratings and measure the accuracy for the ratings, and for the likes we measured accuracy, precision and recall

#### Ratings
"""

print("Accuracy of ratings")
(ratings_test == ratings).astype(int).todense().sum() / (ratings.shape[0]*ratings.shape[1])

# create functions to extract accuracy, precision and recall from matrices of predictions

def total(x):
  return sum(sum(x))

def truePositives(x,y): # x:actual, y: predicted
  tp = np.logical_and(x.copy(), y.copy())
  return tp

def trueNegatives(x,y): # x:actual, y: predicted
  pn = np.logical_not(y.copy())
  nn = np.logical_not(x.copy(), y.copy())
  tn = np.logical_and(pn, nn)
  return tn

def falsePositives(x,y): # x:actual, y: predicted
  n = np.logical_not(x.copy())
  fp = np.logical_and(y.copy(), n)
  return fp

def falseNegatives(x,y): # x:actual, y: predicted
  pn = np.logical_not(y.copy())
  fn = np.logical_and(pn, x.copy())
  return fn

def accuracy(x,y): # x:actual, y: predicted
  return (total(truePositives(x,y)) + total(trueNegatives(x,y))) / (total(x == x))

def precision(x,y): # x:actual, y: predicted
  return total(truePositives(x,y)) / (total(truePositives(x,y)) + total(falsePositives(x,y)))

def recall(x,y): # x:actual, y: predicted
  return total(truePositives(x,y)) / (total(truePositives(x,y)) + total(falseNegatives(x,y)))

# convert the ratings and predicted ratings to binary (like, not like)
likes = ratings.copy() >= 3
likes = np.array(likes.todense())

likes_test = ratings_test.copy() >= 3
likes_test = np.array(likes_test.todense())


likes_pseudo = pseudoRatings.copy() >= 3
likes_pseudo = np.array(likes_pseudo.todense())

"""#### Likes"""

# Testing the pseudo-recommendations
print("Accuracy:", accuracy(likes, likes_pseudo))
print("Precision:", precision(likes, likes_pseudo))
print("Recall:", recall(likes, likes_pseudo))

# Testing the hybrid-recommendations
print("Accuracy:", accuracy(likes, likes_test))
print("Precision:", precision(likes, likes_test))
print("Recall:", recall(likes, likes_test))

