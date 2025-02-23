import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy, PredictionImpossible
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def precision_at_k(predictions, k):
    recommended_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]
    relevant_items = [pred for pred in predictions if pred.r_ui >= 4]
    relevant_recommended = sum(1 for item in recommended_items if item in relevant_items)
    return relevant_recommended / k if k > 0 else 0

def recall_at_k(predictions, k):
    relevant_items = [pred for pred in predictions if pred.r_ui >= 3.5]
    recommended_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]
    relevant_recommended = sum(1 for item in recommended_items if item in relevant_items)
    return relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0

def hit_rate(predictions, k):
    recommended_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:k * 2]
    hits = sum(1 for pred in recommended_items if pred.r_ui >= 3.5)
    return hits / len(predictions) if len(predictions) > 0 else 0

def mean_squared_error_metric(predictions):
    actual_ratings = [pred.r_ui for pred in predictions]
    predicted_ratings = [pred.est for pred in predictions]
    return mean_squared_error(actual_ratings, predicted_ratings)

def train_svd(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userid', 'movieid', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.2, random_state=42)
    
    try:
        param_grid_svd = {'n_factors': [50, 100], 'n_epochs': [20, 30], 
                          'lr_all': [0.007, 0.02], 'reg_all': [0.01, 0.05]}
        gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse', 'fcp'], cv=3)
        gs_svd.fit(data)
        
        best_params = gs_svd.best_params['rmse']
        algo_svd = SVD(n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'],
                       lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], random_state=42)
        algo_svd.fit(trainset)
        predictions_svd = algo_svd.test(testset)
        
        rmse_svd = accuracy.rmse(predictions_svd)
        precision = precision_at_k(predictions_svd, 5)
        recall = recall_at_k(predictions_svd, 5)
        hit_rate_val = hit_rate(predictions_svd, 5)
        mse = mean_squared_error_metric(predictions_svd)
        
        print(f"Precision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")
        print(f"Hit Rate: {hit_rate_val:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        user_features = algo_svd.pu
        item_features = algo_svd.qi
        np.save("user_features3.npy", user_features)
        np.save("item_features3.npy", item_features)

        return algo_svd
    except Exception as e:
        print(f"Error training SVD model: {e}")
        return None

if __name__ == "__main__":
    try:
        ratings_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
        movies_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", header=None, usecols=[0,1], names=['movieid', 'title'])
        users_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.user', sep='|', names=['userid', 'age', 'gender', 'occupation', 'zipcode'])
        
        svd_model = train_svd(ratings_df)
        if svd_model:
            print("SVD model trained successfully.")
        else:
            print("SVD model training failed.")
    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {e}")