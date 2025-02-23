""""SVD but also user personal detailes based"""
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
try:
    ratings_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
    movies = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", header=None, usecols=[0,1, *range(5, 24)], names=['movieid', 'title', *[f'genre_{i}' for i in range(19)]])
    movie_titles = dict(zip(movies['movieid'], movies['title']))
    users = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.user', sep='|', names=['userid', 'age', 'gender', 'occupation', 'zipcode'])
except FileNotFoundError:
    print("Error: One or more data files not found. Please check the file paths.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
ratings_df = pd.merge(ratings_df, movies, on='movieid')
ratings_df = pd.merge(ratings_df, users, on='userid')
ratings_df['gender_encoded'] = ratings_df['gender'].map({'M': 0, 'F': 1})
ratings_df = pd.get_dummies(ratings_df, columns=['occupation'])
ratings_df = pd.get_dummies(ratings_df, columns=[f'genre_{i}' for i in range(19)])
ratings_df['zipcode'] = ratings_df['zipcode'].astype(str)
ratings_df = pd.get_dummies(ratings_df, columns=['zipcode'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userid', 'movieid', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.2, random_state=42)
try:
    param_grid = {'n_factors': [50, 100, 150], 'n_epochs': [20, 30, 40]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'fcp'], cv=3)
    gs.fit(data)
    best_n_factors = gs.best_params['rmse']['n_factors']
    best_n_epochs = gs.best_params['rmse']['n_epochs']
    print(f"Best n_factors: {best_n_factors}")
    print(f"Best n_epochs: {best_n_epochs}")
    model = SVD(n_factors=best_n_factors, n_epochs=best_n_epochs, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE (after tuning): {rmse}")
except Exception as e:
    print(f"An error occurred during model training or evaluation: {e}")
    exit()
def recommend_movies_svd(user_id, top_n=5):
    user_ratings = ratings_df[ratings_df['userid'] == user_id]
    rated_movie_ids = user_ratings['movieid'].tolist()
    unrated_movie_ids = [movie_id for movie_id in movies['movieid'] if movie_id not in rated_movie_ids]
    predictions = []
    for movie_id in unrated_movie_ids:
        try: 
            predictions.append((movie_id, model.predict(user_id, movie_id).est))
        except Exception as e:
            print(f"Error predicting rating for movie {movie_id}: {e}")
            continue 
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, est in predictions[:top_n]]
    print(f"Top {top_n} recommended movies for User {user_id}:")
    for i, movie_id in enumerate(top_movie_ids):
        title = movie_titles.get(movie_id)
        if title:
            print(f"{i+1}. {title} - Predicted Rating: {predictions[i][1]:.2f}")
        else:
            print(f"{i+1}. Movie ID {movie_id} - Predicted Rating: {predictions[i][1]:.2f} (Title not found)")
    return top_movie_ids
target_user_id = 1 
try:
    recommendations = recommend_movies_svd(target_user_id, top_n=5)
except Exception as e:
    print(f"An error occurred during recommendation generation: {e}")

