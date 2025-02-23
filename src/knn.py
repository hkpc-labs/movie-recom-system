import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy, PredictionImpossible
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np

def recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    recalls = []
    for uid, ratings in user_est_true.items():
        relevant_items = sum(1 for _, true_r in ratings if true_r >= threshold)
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        retrieved_relevant = sum(1 for _, true_r in top_k if true_r >= threshold)
        recalls.append(retrieved_relevant / relevant_items if relevant_items else 0)
    return np.mean(recalls)

def mean_avg_precision(predictions, k=10):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    average_precisions = []
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        relevant_count = 0
        sum_precisions = 0
        for i, (_, true_r) in enumerate(ratings[:k]):
            if true_r >= 3.5:
                relevant_count += 1
                sum_precisions += relevant_count / (i + 1)
        if relevant_count > 0:
            average_precisions.append(sum_precisions / relevant_count)
    return np.mean(average_precisions) if average_precisions else 0

def precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    precisions = []
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        relevant = sum(1 for _, true_r in top_k if true_r >= threshold)
        precisions.append(relevant / k if k > 0 else 0)
    return np.mean(precisions)

def train_knn(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userid', 'movieid', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.2, random_state=42)

    try:
        param_grid_knn = {'k': [20, 40, 60], 
                          'sim_options': {'name': ['cosine'], 'user_based': [False]}}
        gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse', 'fcp'], cv=3)
        gs_knn.fit(data)

        best_k_knn = gs_knn.best_params['rmse']['k']

        print(f"Best k (k-NN): {best_k_knn}")

        sim_options = {'name': 'cosine', 'user_based': False}
        algo_knn = KNNBasic(k=best_k_knn, sim_options=sim_options)
        algo_knn.fit(trainset)
        predictions_knn = algo_knn.test(testset)
        rmse_knn = accuracy.rmse(predictions_knn)
        print(f"RMSE (k-NN): {rmse_knn}")

        return algo_knn, testset

    except Exception as e:
        print(f"Error training k-NN model: {e}")
        return None, None

def predict_knn(user_id, movie_id, algo_knn):
    try:
        prediction = algo_knn.predict(user_id, movie_id).est
        return prediction
    except Exception as e:
        print(f"Error predicting with k-NN: {e}")
        return None

def create_user_item_profiles(ratings_df):
    user_profiles = ratings_df.groupby('userid').apply(lambda x: x.set_index('movieid')['rating'].to_dict()).to_dict()
    item_profiles = ratings_df.groupby('movieid').apply(lambda x: x.set_index('userid')['rating'].to_dict()).to_dict()
    return user_profiles, item_profiles

def recommend_movies_knn(user_id, top_n, algo_knn, movies_df, ratings_df):
    try:
        user_ratings = ratings_df[ratings_df['userid'] == user_id]
        rated_movie_ids = user_ratings['movieid'].tolist()
        unrated_movie_ids = [movie_id for movie_id in movies_df['movieid'] if movie_id not in rated_movie_ids]

        predictions_user = []
        for movie_id in unrated_movie_ids:
            try:
                predicted_rating = predict_knn(user_id, movie_id, algo_knn)
                if predicted_rating is not None:
                    predictions_user.append((movie_id, predicted_rating))
            except PredictionImpossible:
                print(f"Cannot predict rating for user {user_id} and movie {movie_id}. Skipping.")
                continue
            except Exception as e:
                print(f"Error predicting rating for movie {movie_id}: {e}")
                continue

        predictions_user.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [movie_id for movie_id, est in predictions_user[:top_n]]

        recommended_movies = movies_df[movies_df['movieid'].isin(top_movie_ids)]

        print(f"Top {top_n} recommended movies for User {user_id} (k-NN):")

        for i, movie_id in enumerate(top_movie_ids):
            movie_title = recommended_movies[recommended_movies['movieid'] == movie_id]['title'].iloc[0]
            predicted_rating = next((rating for mid, rating in predictions_user if mid == movie_id), None)
            if predicted_rating is not None:
                print(f"{i+1}. {movie_title} - Predicted Rating: {predicted_rating:.2f}")

        return top_movie_ids

    except Exception as e:
        print(f"An error occurred during recommendation generation: {e}")
        return []

if __name__ == '__main__':
    try:
        ratings_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
        movies_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", header=None, usecols=[0,1, *range(5, 24)], names=['movieid', 'title', *[f'genre_{i}' for i in range(19)]])

        knn_model, testset = train_knn(ratings_df)

        if knn_model:
            user_id = 1
            top_n = 5
            recommended_movies = recommend_movies_knn(user_id, top_n, knn_model, movies_df, ratings_df)
            if recommended_movies:
                print("\nRecommended movie IDs (k-NN):", recommended_movies)
                predictions_knn = knn_model.test(testset)
                recall = recall_at_k(predictions_knn, k=5)
                map_score = mean_avg_precision(predictions_knn, k=5)
                precision = precision_at_k(predictions_knn, k=5)  # Calculate precision

                print(f"Recall@5: {recall:.4f}")
                print(f"Mean Average Precision@5: {map_score:.4f}")
                print(f"Precision@5: {precision:.4f}")  # Print precision

        else:
            print("k-NN model training failed. Cannot make recommendations.")

    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths")