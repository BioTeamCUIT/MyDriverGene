"""
Read the genes in the KEGG pathways and construct pathway co-occurrence
The gene co-occurrence relationship of two gene was calculated using cosine similarity
"""
import os
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.utils import from_scipy_sparse_matrix

# os.chdir("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes")
os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


class PathwayConstructor():
    def __init__(self, file_path, thr_path=0.5):
        super(PathwayConstructor, self).__init__()
        self.file_path = file_path
        self.thr_path = thr_path


    # def read_genes(self):
    #     # path_list = os.listdir(self.file_path)
    #     all_genes_set = set()
    #     file_names = [file for file in os.listdir(self.file_path) if file.endswith('.csv')]
    #     for file_name in file_names:
    #         temp_df = pd.read_csv(os.path.join(self.file_path, file_name), header=None)
    #         all_genes_set = all_genes_set | set(temp_df.values[0])
    #     return all_genes_set
    #
    # def calculate_co_occurrence(self):
    #     genes = self.read_genes()
    #     genes = pd.DataFrame(genes, columns=['GeneList'])
    #     file_names = [file for file in os.listdir(self.file_path) if file.endswith('.csv')]
    #     for file_name in file_names:
    #         temp_df = pd.read_csv(os.path.join(self.file_path, file_name), header=None)
    #         temp_df = temp_df.T
    #         tag = np.ones(shape=temp_df.shape[0])

    def construct_pathway(self):
        print("Reading KEGG, please waite ... \n")
        kegg = pd.read_csv(os.path.join(self.file_path, "path_similarity.csv"), index_col=0)
        # kegg = pd.read_csv(os.path.join(self.file_path, "pathsim_matrix.csv"), index_col=0, sep='\t')
        # Get the gene with omic feature
        omic_feature = pd.read_csv("RawData/OmicFeatures/biological_features.csv", sep='\t', index_col=0)
        # kegg_genes: 8200
        kegg_genes = set(kegg.columns.values)
        # omic_gene: 13627
        omic_gene = set(omic_feature.index.values)
        # genes_keep: 6585
        genes_keep = list(kegg_genes & omic_gene)
        kegg = kegg.loc[genes_keep][genes_keep]

        # Filter interactions by threshold
        kegg_matrix = kegg.applymap(lambda x: 0 if x < self.thr_path else 1)
        np.fill_diagonal(kegg_matrix.values, 0)
        # Construct edge_index from adj
        edge_index_pathway = from_scipy_sparse_matrix(sp.coo_matrix(kegg_matrix))[0]

        unique_genes = pd.DataFrame(kegg.columns.values).sort_values(by=0)
        # Reorganize gene id
        unique_gene_id = pd.DataFrame(data={
            'GeneSymbol': unique_genes[0].unique(),
            'MappedGeneID': pd.RangeIndex(unique_genes.shape[0])
        })

        return edge_index_pathway, unique_gene_id

# if __name__ == '__main__':
#     file_path = "RawData\\Associations\\KEGG"
#     constructor = PathwayConstructor(file_path=file_path)
#     pathway_co_occurrence = constructor.calculate_co_occurrence()
#     print()
