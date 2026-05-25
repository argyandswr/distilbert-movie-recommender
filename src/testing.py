import torch
import os
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings

# Suppress transformer warnings to keep the terminal output clean
warnings.filterwarnings("ignore")

def load_system():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../models/best_model_v3')
    mlb_path = os.path.join(current_dir, '../mlb_binarizer.pkl') 
    
    print("Loading model and tokenizer... 🚀")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    print("Loading MultiLabelBinarizer (MLB)... 🗂️")
    mlb = joblib.load(mlb_path)
    
    return tokenizer, model, mlb

def predict(text, tokenizer, model, mlb):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=256
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Apply sigmoid activation to obtain probabilities between 0 and 1
    probs = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    preds = (probs > 0.25).astype(int)
    
    try:
        predicted_genres = mlb.inverse_transform(preds.reshape(1, -1))[0]
    except Exception as e:
        predicted_genres = []
        
    return predicted_genres, probs

def main():
    tokenizer, model, mlb = load_system()
    
    print("\n" + "="*40)
    print("🎬 MOVIE GENRE PREDICTION SYSTEM 🎬")
    print("Type 'exit' to terminate the program")
    print("="*40 + "\n")
    
    while True:
        synopsis = input("📝 Enter movie synopsis: ")
        
        if synopsis.lower() == 'exit':
            print("Exiting program... 🏃‍♂️")
            break
            
        genres, probs = predict(synopsis, tokenizer, model, mlb)
        
        result = ', '.join(genres) if genres else 'Unclassified (Confidence below threshold)'
        print(f"\n🎯 Predicted Genre(s) (Threshold > 0.25): {result}")
        
        print("\n🔍 Raw Probabilities (Top 10 Highest Scores):")
        genre_probs = list(zip(mlb.classes_, probs))
        genre_probs.sort(key=lambda x: x[1], reverse=True)
        
        for genre, prob in genre_probs[:10]:
            print(f"  - {genre}: {prob:.4f} ({prob*100:.1f}%)")
            
        print("-" * 50)

if __name__ == "__main__":
    main()