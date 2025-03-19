import rdkit
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Constants:
    BOND_TYPES: list = field(
        default_factory=lambda:
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ]
        )
    BOND_DIRS: list = field(
        default_factory=lambda:
        [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    )


def mol_to_graph(mol: rdkit.Chem.Mol) -> torch_geometric.data.Data:
    const = Constants()
    nodes_list: list = []
    edge_index: list = []
    edge_attr: list = []

    for atom in mol.GetAtoms():
        nodes_list.append([])
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_feature = [const.BOND_TYPES.index(bond.GetBondType()),
                        const.BOND_DIRS.index(bond.GetBondDir())]
        edge_index.append([i, j])
        edge_attr.append(edge_feature)
        edge_index.append([j, i])
        edge_attr.append(edge_feature)