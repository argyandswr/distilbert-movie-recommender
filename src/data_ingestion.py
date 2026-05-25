import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the complete dataset from the specified CSV file path without sampling.
    """
    print("Loading the complete dataset...")
    df_full = pd.read_csv(file_path)
    return df_full