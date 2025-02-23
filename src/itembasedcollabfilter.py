import pandas as pd
import numpy as np
import gradio as gr
from surprise import Dataset, Reader, SVD, accuracy, PredictionImpossible
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

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

def train_svd(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userid', 'movieid', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.2, random_state=42)
    
    param_grid_svd = {'n_factors': [50, 100], 'n_epochs': [20, 30], 
                      'lr_all': [0.007, 0.02], 'reg_all': [0.01, 0.05]}
    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
    gs_svd.fit(data)
    
    best_params = gs_svd.best_params['rmse']
    algo_svd = SVD(n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'],
                   lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], random_state=42)
    algo_svd.fit(trainset)
    return algo_svd

def train_item_based(ratings_df):
    train_matrix = ratings_df.pivot(index='userid', columns='movieid', values='rating').fillna(0)
    train_sparse = csr_matrix(train_matrix.values)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(train_sparse.T)
    return model, train_matrix

def predict_rating_item_based(user_id, movie_id, model, train_matrix):
    if movie_id not in train_matrix.columns:
        return train_matrix.mean().mean()
    movie_index = train_matrix.columns.get_loc(movie_id)
    distances, indices = model.kneighbors(train_matrix.values.T[movie_index].reshape(1, -1), n_neighbors=6)
    weighted_sum, similarity_sum = 0, 0
    for i in range(1, len(indices[0])):
        neighbor_index_sparse = indices[0][i]
        similarity = 1 - distances[0][i]
        neighbor_movie_id = train_matrix.columns[neighbor_index_sparse]
        neighbor_rating = train_matrix.loc[user_id, neighbor_movie_id]
        weighted_sum += similarity * neighbor_rating
        similarity_sum += similarity
    return weighted_sum / similarity_sum if similarity_sum else train_matrix.mean().mean()

def recommend_movies(userid, model_type, top_n=10):
    ratings_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.data', sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
    movies_df = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding="latin-1", header=None, usecols=[0,1], names=['movieid', 'title'])
    movie_titles = dict(zip(movies_df['movieid'], movies_df['title']))
    
    if model_type == 'SVD':
        model = train_svd(ratings_df)
        if not model:
            return ["Model training failed"]
        user_ratings = np.dot(model.pu[int(userid) - 1], model.qi.T)
        top_movie_indices = np.argsort(user_ratings)[::-1][:top_n]
    else:
        model, train_matrix = train_item_based(ratings_df)
        rated_movies = ratings_df[ratings_df['userid'] == userid]['movieid'].tolist()
        unrated_movies = [movie_id for movie_id in train_matrix.columns if movie_id not in rated_movies]
        predicted_ratings = [predict_rating_item_based(userid, movie_id, model, train_matrix) for movie_id in unrated_movies]
        top_movie_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        unrated_movies = np.array(unrated_movies)
        top_movie_indices = unrated_movies[top_movie_indices]
    
    recommendations = [movie_titles.get(idx, "Unknown Movie") for idx in top_movie_indices]
    return recommendations

with gr.Blocks() as app:
    gr.Markdown("# 🎬 Movie Recommendation System")
    gr.Markdown("Select a model and get personalized movie recommendations.")
    user_id_input = gr.Number(label="Enter User ID", value=1, minimum=1, maximum=943)
    model_selector = gr.Radio(["SVD", "Item-Based"], label="Select Recommendation Model", value="SVD")
    num_recs_slider = gr.Slider(1, 10, step=1, label="Number of Recommendations")
    rec_button = gr.Button("Get Recommendations")
    rec_output = gr.Textbox(label="Recommended Movies", interactive=False, lines=10)
    rec_button.click(fn=recommend_movies, inputs=[user_id_input, model_selector, num_recs_slider], outputs=rec_output)

app.launch(share=True)
