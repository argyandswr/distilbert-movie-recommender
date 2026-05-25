import os
import joblib
import pandas as pd
import numpy as np

# Set path biar konsisten
current_dir = os.path.dirname(os.path.abspath(__file__))
mlb_path = os.path.join(current_dir, '../mlb_binarizer.pkl')
data_path = os.path.join(current_dir, '../datasets/processed/movies_balanced.csv')

# 1. Inspeksi MLB Classes
print("🔍 === CEK URUTAN MLB ===")
mlb = joblib.load(mlb_path)
print(f"Total Genre: {len(mlb.classes_)}")
print(mlb.classes_)

# 2. Inspeksi Pos Weights
print("\n⚖️ === CEK POS WEIGHTS ===")
df = pd.read_csv(data_path)
df['genre_list'] = df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

y = mlb.transform(df['genre_list'])

# Hitung manual bobot per kelas
num_positives = np.sum(y, axis=0)
num_negatives = len(y) - num_positives
pos_weights = num_negatives / (num_positives + 1e-5)

print(f"{'GENRE':<17} | {'WEIGHT':<8} | {'JUMLAH DATA'}")
print("-" * 45)
for genre, weight, count in zip(mlb.classes_, pos_weights, num_positives):
    print(f"{genre:<17} | {weight:<8.2f} | {count}")