import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Prepare the OHE of the strains
def prep_ohe(categories):

    """
    Prepare one-hot encoding for strain variables.

    This function creates a one-hot encoding representation of the provided categorical variables.
    It fits a OneHotEncoder to the categories and transforms them into a pandas DataFrame.

    Parameters:
    - categories (array-like): Array-like object containing categorical variables.

    Returns:
    - cat_ohe (pandas.DataFrame): DataFrame representing the one-hot encoded categorical variables.
    """
    # Prepare OHE
    ohe = OneHotEncoder(sparse_output=False)

    # Fit OHE
    ohe.fit(pd.DataFrame(categories))

    # Prepare OHE
    cat_ohe = pd.DataFrame(ohe.transform(pd.DataFrame(categories)), columns=categories, index=categories)

    return cat_ohe

def add_strains(chemfeats_df, screen_path):

    """
    Add strains to chemical features using Cartesian product merge.

    This function adds strains to chemical features using Cartesian product merge
    between the chemical features DataFrame and the one-hot encoded strains DataFrame.

    Parameters:
    - chemfeats_df (pandas.DataFrame): DataFrame containing chemical features.
    - screen_path (str): Path to the maier_screening_results.tsv file.

    Returns:
    - xpred (pandas.DataFrame): DataFrame containing chemical features with added strains.
    """
    # Read screen information and One-hot-encode strains
    maier_screen = pd.read_csv(screen_path, sep='\t', index_col=0)
    ohe_df = prep_ohe(maier_screen.columns)

     # Prepare chemical features
    chemfe = chemfeats_df.reset_index().rename(columns={"index": "chem_id"})
    chemfe["chem_id"] = chemfe["chem_id"].astype(str) 

    # Prepare OHE
    sohe = ohe_df.reset_index().rename(columns={"index": "strain_name"})

    # Cartesian product merge
    xpred = chemfe.merge(sohe, how="cross")
    xpred["pred_id"] = xpred["chem_id"].str.cat(xpred["strain_name"], sep=":")

    xpred = xpred.set_index("pred_id")
    xpred = xpred.drop(columns=["chem_id", "strain_name"])

    # Make sure correct number of rows
    assert xpred.shape[0] == (chemfeats_df.shape[0] * ohe_df.shape[0])

    # Make sure correct number of features
    assert xpred.shape[1] == (chemfeats_df.shape[1] + ohe_df.shape[1])
    
    return xpred

if __name__ == "__main__":
    #main()
    pass