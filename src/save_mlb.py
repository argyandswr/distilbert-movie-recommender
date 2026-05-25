import os
import joblib
import pandas as pd
from preprocessing import preprocess_data

# Load dataset lo
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../datasets/processed/movies_dataset_ready.csv')
    
# Loading the entire dataset
df = pd.read_csv(data_path)

# Panggil fungsi andalan lo
_, _, _, _, _, mlb = preprocess_data(df)

# Save mlb-nya
joblib.dump(mlb, 'mlb_binarizer.pkl')
print("MLB Saved")