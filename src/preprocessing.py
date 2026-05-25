import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_data(df):
    """
    Process the genres column and split the dataset into training and testing sets.
    """
    df['genre_list'] = df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['genre_list'])
    num_labels = len(mlb.classes_)

    X = df['overview'].to_numpy()
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, num_labels, mlb