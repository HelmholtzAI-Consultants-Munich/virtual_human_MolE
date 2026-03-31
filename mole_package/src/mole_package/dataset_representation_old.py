import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset, Batch

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

# Here we can add more molecular descriptors
class MoleculeDataset(Dataset):

    """
    Dataset class for creating molecular graphs.

    Attributes:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - smile_column (str): Name of the column containing SMILES strings.
    - id_column (str): Name of the column containing molecule IDs.
    """

    def __init__(self, smile_df, smile_column, id_column):
        super(Dataset, self).__init__()

        # Gather the SMILES and the corresponding IDs
        self.smiles_data = smile_df[smile_column].tolist()
        self.id_data = smile_df[id_column].tolist()

    def __getitem__(self, index):
        # Get the molecule
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        chirality_idx = []
        atomic_number = []

        # Roberto: Might want to add more features later on. Such as atomic spin
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                print(self.id_data[index])

            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    chem_id=self.id_data[index])
        
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()

# Function to generate the molecular representation with MolE
def batch_representation(smile_df, dl_model, column_str, id_str, batch_size= 10_000, id_is_str=True, device="cuda:0"):

    """
    Generate molecular representations using a Deep Learning model.

    Parameters:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - dl_model: Deep Learning model for molecular representation.
    - column_str (str): Name of the column containing SMILES strings.
    - id_str (str): Name of the column containing molecule IDs.
    - batch_size (int, optional): Batch size for processing (default is 10,000).
    - id_is_str (bool, optional): Whether IDs are strings (default is True).
    - device (str, optional): Device for computation (default is "cuda:0").

    Returns:
    - chem_representation (pandas.DataFrame): DataFrame containing molecular representations.
    """
    
    # First we create a list of graphs
    molecular_graph_dataset = MoleculeDataset(smile_df, column_str, id_str)
    graph_list = [g for g in molecular_graph_dataset]

    # Determine number of loops to do given the batch size
    n_batches = len(graph_list) // batch_size

    # Are all molecules accounted for?
    remaining_molecules = len(graph_list) % batch_size

    # Starting indices
    start, end = 0, batch_size

    # Determine number of iterations
    if remaining_molecules == 0:
        n_iter = n_batches
    
    elif remaining_molecules > 0:
        n_iter = n_batches + 1
    
    # A list to store the batch dataframes
    batch_dataframes = []

    # Iterate over the batches
    for i in range(n_iter):
        # Start batch object
        batch_obj = Batch()
        graph_batch = batch_obj.from_data_list(graph_list[start:end])
        graph_batch = graph_batch.to(device)

        # Gather the representation
        with torch.no_grad():
            dl_model.eval()
            h_representation, _ = dl_model(graph_batch)
            chem_ids = graph_batch.chem_id
        
        batch_df = pd.DataFrame(h_representation.cpu().numpy(), index=chem_ids)
        batch_dataframes.append(batch_df)

        # Get the next batch
        ## In the final iteration we want to get all the remaining molecules
        if i == n_iter - 2:
            start = end
            end = len(graph_list)
        else:
            start = end
            end = end + batch_size
    
    # Concatenate the dataframes
    chem_representation = pd.concat(batch_dataframes)

    return chem_representation