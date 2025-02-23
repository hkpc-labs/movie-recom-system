import os
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ratings = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
movies = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", usecols=[0,1], names=['movieid', 'title'])

movie_titles = dict(zip(movies['movieid'], movies['title']))

num_users = ratings['userid'].nunique()
num_movies = ratings['movieid'].nunique()
num_ratings = len(ratings)

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Number of ratings: {num_ratings}")
print(f"Sparsity: {1 - (num_ratings / (num_users * num_movies))}")

user_item_matrix = ratings.pivot(index='userid', columns='movieid', values='rating').fillna(0)

nmf = NMF(n_components=20, init='random', random_state=42, max_iter=100000)
user_features = nmf.fit_transform(user_item_matrix)
item_features = nmf.components_

predicted_ratings_matrix = np.dot(user_features, item_features) 

def recommend_movies(user_id, user_features, item_features, movie_titles, top_n=15):
    user_ratings = np.dot(user_features[user_id], item_features)
    top_movie_indices = np.argsort(user_ratings)[::-1][:top_n]
    print(f"Top {top_n} recommended movies for User {user_id+1}:") 
    for i, idx in enumerate(top_movie_indices):
        print(f"{i+1}. {movie_titles[idx+1]} - Predicted Rating: {user_ratings[idx]:.2f}") 
    return top_movie_indices

user_id = 10
top_movie_indices = recommend_movies(user_id=user_id, user_features=user_features, item_features=item_features, movie_titles=movie_titles)

actual_ratings_list = []
predicted_ratings_list = []

for index, row in ratings.iterrows():
    user_idx = int(row['userid']) - 1 
    movie_idx = int(row['movieid']) - 1 
    actual_rating = row['rating']
    
    if 0 <= user_idx < user_features.shape[0] and 0 <= movie_idx < item_features.shape[1]: 
        predicted_rating = predicted_ratings_matrix[user_idx, movie_idx]
        actual_ratings_list.append(actual_rating)
        predicted_ratings_list.append(predicted_rating)

actual_ratings_array = np.array(actual_ratings_list)
predicted_ratings_array = np.array(predicted_ratings_list)

if len(actual_ratings_array) > 0 and len(predicted_ratings_array) > 0: 
    rmse = np.sqrt(mean_squared_error(actual_ratings_array, predicted_ratings_array))
    print(f"RMSE: {rmse:.4f}")
else:
    print("Error: No valid ratings found for RMSE calculation.")
 
np.save('user_features1.npy', user_features)
np.save('item_features1.npy', item_features)

def precision_recall_at_k(user_id, top_movie_indices, actual_ratings, k=10):
    relevant_items = set(actual_ratings[actual_ratings['userid'] == user_id + 1]['movieid'])
    recommended_items = set(top_movie_indices[:k] + 1)
    
    num_relevant = len(relevant_items & recommended_items)
    precision = num_relevant / k
    recall = num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0
    print(f"User {user_id+1} - Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")
    return precision, recall

precision_recall_at_k(user_id, top_movie_indices, ratings, k=10)

def mean_average_precision(actual_ratings, predicted_ratings_matrix, k=10):
    map_total = 0
    for user_id in range(num_users):
        relevant_items = set(actual_ratings[actual_ratings['userid'] == user_id + 1]['movieid'])
        user_ratings = predicted_ratings_matrix[user_id]
        top_movie_indices = np.argsort(user_ratings)[::-1][:k] + 1
        
        num_relevant = len(relevant_items & set(top_movie_indices))
        precision = num_relevant / k
        map_total += precision
    
    map_score = map_total / num_users
    print(f"Mean Average Precision (MAP)@{k}: {map_score:.4f}")
    return map_score

mean_average_precision(ratings, predicted_ratings_matrix, k=10)
