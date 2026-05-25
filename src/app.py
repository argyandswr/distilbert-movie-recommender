import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ---------------------------------------------------------------------
# 1. CONFIGURATION & RESOURCE LOADING
# ---------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

@st.cache_resource
def load_ml_assets():
    """
    Load and cache the fine-tuned DistilBERT model, tokenizer, and MLB.
    """
    # Adjust paths based on your actual repository architecture
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "notebooks" in os.getcwd() else os.getcwd()
    model_path = os.path.join(base_dir, "models/best_model_v3")
    mlb_path = os.path.join(base_dir, "mlb_binarizer.pkl")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    
    mlb = joblib.load(mlb_path)
    return tokenizer, model, mlb

@st.cache_data
def load_catalog_data():
    """
    Load the reference dataset for running recommendation queries.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "notebooks" in os.getcwd() else os.getcwd()
    data_path = os.path.join(base_dir, "datasets/processed/movies_40k.csv")
    df = pd.read_csv(data_path)
    return df

# Initialize assets
try:
    tokenizer, model, mlb = load_ml_assets()
    movie_catalog = load_catalog_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 2. INFERENCE & RECOMMENDATION LOGIC
# ---------------------------------------------------------------------
def predict_genres(synopsis, threshold=0.35):
    """
    Run inference on the input text to extract predicted genre tags.
    """
    inputs = tokenizer(
        synopsis,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Filter genres based on the specified threshold
    predicted_indices = np.where(probs > threshold)[0]
    if len(predicted_indices) == 0:
        predicted_indices = np.array([np.argmax(probs)]) # Fallback to top-1 if none pass threshold
        
    predicted_genres = [mlb.classes_[idx] for idx in predicted_indices]
    return predicted_genres, probs

def get_recommendations(predicted_genres, catalog, top_n=5):
    """
    Query the catalog for movies sharing overlapping genres, sorted by ranking metrics.
    """
    # Ensure the genre_list column exists in the dataframe context
    if 'genre_list' not in catalog.columns:
        catalog['genre_list'] = catalog['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
        
    def calculate_intersection(movie_genres):
        return len(set(predicted_genres).intersection(set(movie_genres)))
    
    # Calculate matching score
    catalog['match_score'] = catalog['genre_list'].apply(calculate_intersection)
    
    # Filter and sort by match score, then by popularity/rating
    recommendations = catalog[catalog['match_score'] > 0].sort_values(
        by=['match_score', 'vote_average', 'popularity'], 
        ascending=[False, False, False]
    )
    
    return recommendations.head(top_n)

# ---------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE UI
# ---------------------------------------------------------------------
st.title("🎬 Intelligent Movie Recommendation System")
st.markdown("Enter a movie synopsis below. The fine-tuned **DistilBERT** model will predict its genres and recommend similar films from our database.")

# Layout column splitting
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Synopsis")
    user_synopsis = st.text_area(
        label="Paste movie plot summary here:",
        height=200,
        placeholder="Type or paste the storyline..."
    )
    
    target_threshold = st.slider("Classification Probability Threshold", 0.10, 0.90, 0.35, step=0.05)
    submit_button = st.button("Generate Recommendations", type="primary")

with col2:
    st.subheader("Model Analysis Output")
    if submit_button and user_synopsis.strip():
        with st.spinner("Executing model inference pipeline..."):
            # Run prediction
            genres, probabilities = predict_genres(user_synopsis, target_threshold)
            
            # Display Tag Badges
            st.write("**Predicted Genres:**")
            genre_badges = "".join([f"`{g}` " for g in genres])
            st.markdown(genre_badges)
            
            # Show top-5 probability chart
            st.write("**Prediction Confidence:**")
            top_indices = np.argsort(probabilities)[-5:][::-1]
            chart_data = pd.DataFrame({
                "Genre": [mlb.classes_[i] for i in top_indices],
                "Confidence": [probabilities[i] for i in top_indices]
            })
            st.bar_chart(data=chart_data, x="Genre", y="Confidence")
    else:
        st.info("Awaiting input data. Paste a synopsis and trigger execution.")

# Display Recommendations Section below the inputs
if submit_button and user_synopsis.strip():
    st.separator()
    st.subheader("🎯 Top Content Recommendations")
    
    results = get_recommendations(genres, movie_catalog)
    
    if not results.empty:
        for idx, row in results.iterrows():
            with st.container():
                st.markdown(f"### {row['title']} ({str(row['release_date'])[:4]})")
                st.markdown(f"**Genres:** `{row['genres']}` | **Rating:** ⭐ {row['vote_average']} | **Popularity Score:** 📈 {row['popularity']}")
                st.write(row['overview'])
                st.markdown("---")
    else:
        st.warning("No overlapping genres discovered in the reference dataset catalog.")