import gradio as gr
import numpy as np
import pandas as pd

model_info = {
"NMF": "📌 *NMF (Non-Negative Matrix Factorization)*:\n - Works well with sparse data.\n - Decomposes user-item matrix into two non-negative matrices.\n - Useful for implicit feedback.",
"SVD": "📌 *SVD (Singular Value Decomposition)*:\n - Captures latent relationships between users and items.\n - Works best when the dataset is dense.\n - Used in famous recommendation systems like Netflix.",
"ALS": "📌 *ALS (Alternating Least Squares)*:\n - Optimized for large-scale recommendation systems.\n - Best when dealing with implicit feedback.\n - Used by Spark ML for scalable recommendations.",
"Hybrid": "📌 *Hybrid Model (Combining NMF & Content-Based Filtering)*:\n - Uses both collaborative filtering and movie metadata.\n - Helps solve the 'Cold Start Problem' for new users."
}

def choose_model(model_name):
    if model_name == 'NMF':
        user_features = np.load('user_features2.npy', allow_pickle=True)
        item_features = np.load('item_features2.npy', allow_pickle=True)
        return user_features, item_features
    elif model_name == 'Hybrid':
        user_features = np.load('user_features1.npy', allow_pickle=True)
        item_features = np.load('item_features1.npy', allow_pickle=True)
        return user_features, item_features
    elif model_name == 'SVD 2.0':
        user_features = np.load('user_features3.npy', allow_pickle=True)
        item_features = np.load('item_features3.npy', allow_pickle=True)
        return user_features, item_features
    
 

movies = pd.read_csv(r'C:\Users\hardi\Downloads\recomendations system\movie-reco\data\ml-100k\u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['movieid', 'title'])
movie_titles = dict(zip(movies["movieid"], movies["title"]))

def recommend_movies(userid, model_name, top_n = 15):
    userid = int(userid) - 1
    user_features, item_features = choose_model(model_name)
    if user_features is None or item_features is None: # Check for load errors
        return ["Error: Could not load features for selected model. Please check the console for details."]

    if userid < 0 or userid >= user_features.shape[0]: # Check if valid user id
        return ["Invalid user ID. Please enter a valid ID."]
        
    user_ratings = np.clip(np.dot(user_features[userid], item_features.T), 1, 5)  # Transpose item_features
    top_movie_indices = np.argsort(user_ratings)[::-1][:top_n]
    recommendations = [movie_titles.get(idx + 1, "Unknown Movie") for idx in top_movie_indices]
    return recommendations
def get_model_info(model_name):
    return model_info.get(model_name, "ℹ Select a model to view details.")


with gr.Blocks() as app:
    gr.Markdown("# 🎬 Movie Recommendation System")
    gr.Markdown("Select a recommendation model and get personalized movie recommendations.")
    model_dropdown = gr.Dropdown(["NMF", "SVD", "SVD 2.0", "KNN", "Hybrid"], label="Select Model", value="NMF")
    model_info_box = gr.Textbox(value=get_model_info("NMF"), label="Model Information", interactive=False)
    model_dropdown.change(fn=get_model_info, inputs=model_dropdown, outputs=model_info_box)
    user_id_input = gr.Number(label="Enter User ID", value=1, minimum=1, maximum=943)
    num_recs_slider = gr.Slider(1, 10, step=1, label="Number of Recommendations")
    rec_button = gr.Button("Get Recommendations")
    rec_output = gr.Textbox(label="Recommended Movies", interactive=False, lines=10)
    rec_button.click(fn=recommend_movies, inputs=[user_id_input, model_dropdown, num_recs_slider], outputs=rec_output)

app.launch(share=True)
