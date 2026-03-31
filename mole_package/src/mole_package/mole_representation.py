import pandas as pd
from rdkit import Chem
import os

# A FUNCTION TO READ SMILES from file 
def read_smiles_df(data_path, smile_col="rdkit_no_salt", id_col="prestwick_ID"):

    """
    Read SMILES data from a file and remove invalid SMILES.

    Parameters:
    - data_path (str): Path to the file containing SMILES data.
    - smile_col (str, optional): Name of the column containing SMILES strings (default is "rdkit_no_salt").
    - id_col (str, optional): Name of the column containing molecule IDs (default is "prestwick_ID").

    Returns:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data with specified columns.
    """
    
    # Read the data
    smile_df = pd.read_csv(data_path, sep='\t')
    smile_df = smile_df[[smile_col, id_col]]

    # Make sure ID column is interpreted as str
    smile_df[id_col] = smile_df[id_col].astype(str)

    # Remove NaN
    smile_df = smile_df.dropna()

    # Remove invalid smiles
    smile_df = smile_df[smile_df[smile_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    return smile_df

def read_smiles(data_path):

    """
    Read SMILES data from a file and remove invalid SMILES.

    Parameters:
    - data_path (str): Path to the file containing SMILES data.
    - smile_col (str, optional): Name of the column containing SMILES strings (default is "rdkit_no_salt").
    - id_col (str, optional): Name of the column containing molecule IDs (default is "prestwick_ID").

    Returns:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data with specified columns.
    """
    
    # Read the data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find file: {data_path}")

    with open(data_path, 'r') as f:
        # Read lines, strip whitespace, and split to handle optional IDs
        # (e.g., 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine')
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]

    # Convert to Mol objects
    # Note: Invalid SMILES will result in None in this list
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Optional: Filter out Nones if you only want valid molecules
    mols = [m for m in mols if m is not None]
    
    return mols

if __name__ == "__main__":
    print(read_smiles("sequences.smiles"))


    
