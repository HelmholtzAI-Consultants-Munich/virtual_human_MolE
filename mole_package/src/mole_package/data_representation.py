import numpy as np
import torch
from torch_geometric.data import Data, Dataset, Batch

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1, 119))
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


class MoleculeDataset(Dataset):
    """
    Dataset class for creating molecular graphs from a list of SMILES strings.
    """

    def __init__(self, smiles_list, id_list=None):
        super().__init__()
        self.smiles_data = smiles_list
        self.id_data = id_list if id_list is not None else list(range(len(smiles_list)))

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        type_idx = []
        chirality_idx = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                print(self.id_data[index])

            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            feat = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            edge_feat.append(feat)
            edge_feat.append(feat)

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            chem_id=self.id_data[index]
        )

    def __len__(self):
        return len(self.smiles_data)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()


def batch_representation(
    smiles_list,
    dl_model,
    batch_size=10_000,
    ids_list=None,
    device="cuda:0"
):
    """
    Generate molecular representations using a Deep Learning model.

    Parameters:
    - smiles_list: list of SMILES strings
    - dl_model: Deep Learning model for molecular representation
    - batch_size: batch size for processing
    - ids_list: optional list of molecule IDs
    - device: device for computation

    Returns:
    - chem_representation: torch.Tensor containing all embeddings
    - chem_ids: list of molecule IDs in the same order
    """

    molecular_graph_dataset = MoleculeDataset(smiles_list, ids_list)
    graph_list = [g for g in molecular_graph_dataset]

    batch_representations = []
    chem_ids_all = []

    for start in range(0, len(graph_list), batch_size):
        end = min(start + batch_size, len(graph_list))
        graph_batch = Batch.from_data_list(graph_list[start:end]).to(device)

        with torch.no_grad():
            dl_model.eval()
            h_representation, _ = dl_model(graph_batch)
            chem_ids = graph_batch.chem_id

        batch_representations.append(h_representation.cpu())
        chem_ids_all.extend(chem_ids)

    chem_representation = torch.cat(batch_representations, dim=0)
    return chem_representation, chem_ids_all