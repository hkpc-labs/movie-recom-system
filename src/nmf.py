import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

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
        precisions.append(relevant / k)
    return np.mean(precisions)

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

def hit_rate(predictions, k=10, threshold=3.5):
    hits = 0
    total = 0
    for uid, _, true_r, est, _ in predictions:
        if true_r >= threshold:
            total += 1
            if est >= threshold:
                hits += 1
    return hits / total if total else 0

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

def f1_score_at_k(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

if __name__ == '__main__':
    try:
        ratings_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
        movies_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", header=None, usecols=[0,1], names=['movieid', 'title'])
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['userid', 'movieid', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=.2, random_state=42)
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30], 'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'fcp'], cv=2, n_jobs=-1)
        gs.fit(data)
        best_params = gs.best_params['rmse']
        algo = SVD(n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], random_state=42)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        precision = precision_at_k(predictions, k=5)
        recall = recall_at_k(predictions, k=5)
        hr = hit_rate(predictions, k=5)
        map_score = mean_avg_precision(predictions, k=5)
        f1 = f1_score_at_k(precision, recall)
        print(f"RMSE: {rmse:.4f}")
        print(f"Precision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")
        print(f"Hit Rate: {hr:.4f}")
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        user_id = 1 
        user_predictions = [pred for pred in predictions if pred.uid == user_id]
        top_movie_predictions = sorted(user_predictions, key=lambda x: x.est, reverse=True)[:5]
        print("Top 5 Recommended Movies:")
        for idx, pred in enumerate(top_movie_predictions):
            movie_title = movies_df[movies_df['movieid'] == pred.iid]['title'].iloc[0]
            print(f"{idx+1}. {movie_title} ")
    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {e}")

user_features = algo.pu
item_features = algo.qi
np.save('user_features2.npy', user_features)
np.save('item_features2.npy', item_features)
