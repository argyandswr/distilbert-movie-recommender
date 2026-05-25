import os
from data_ingestion import load_data
from preprocessing import preprocess_data
from model import train_and_evaluate

def main():
    """
    Main execution script to orchestrate data ingestion, preprocessing, and model training.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../datasets/processed/movies_40k.csv')
    
    # Loading the entire dataset
    df = load_data(data_path)
    print(f"Total records loaded: {len(df)}")
    
    X_train, X_test, y_train, y_test, num_labels, mlb = preprocess_data(df)
    
    train_and_evaluate(X_train, X_test, y_train, y_test, num_labels)

if __name__ == "__main__":
    main()