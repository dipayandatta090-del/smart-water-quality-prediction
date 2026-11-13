import pandas as pd

def load_dataset(path):
    """Load a CSV dataset and remove completely empty rows/columns."""
    df = pd.read_csv(path)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df

def prepare_features_target(df, target_column):
    """Return numeric input features X and target y."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # This starter project expects numeric model inputs.
    X = X.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found.")

    # Remove rows where the target is missing.
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    return X, y
