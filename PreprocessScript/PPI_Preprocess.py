"""
Process Protein-Protein Interactions
"""
import os
import torch
import pandas as pd
from tqdm import tqdm

# os.chdir("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes")
os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


class PPIConstructor():
    def __init__(self, file_path, ppi_type):
        super(PPIConstructor, self).__init__()
        self.file_path = file_path
        self.ppi_type = ppi_type

    def construct_ppi(self):
        print("Reading and constructing PPI, please wait ... \n")
        if self.ppi_type == 'STRING':
            ppi = pd.read_table(os.path.join(self.file_path, "9606.protein.links.full.v11.5.txt"), sep=" ")
            columns_keep = ['protein1', 'protein2', 'combined_score']
            ppi = ppi[columns_keep]
            # Keep interactions whose combined score >= 700
            ppi = ppi[ppi['combined_score'] >= 700]
            # Rename protein id, remove "9606."
            # print("Renaming protein ID, please waite ... \n")
            # pbar = tqdm(range(ppi.shape[0]))
            # for i in pbar:
            #     ppi.iloc[i, 0] = ppi.iloc[i, 0].replace('9606.', '')
            #     ppi.iloc[i, 1] = ppi.iloc[i, 1].replace('9606.', '')
            ppi['protein1'] = ppi['protein1'].str.replace('9606.', '')
            ppi['protein2'] = ppi['protein2'].str.replace('9606.', '')
        elif self.ppi_type == 'CPDB':
            CPDB = pd.read_csv(os.path.join('/private/xiongshuwen/CancerGene/Data/ppi/', "CPDB" + '.tsv'), sep='\t',
                               compression='gzip', encoding='utf8', usecols=['partner1', 'partner2'])
            ensemble = pd.read_csv("RawData/Associations/Ensembl/EnsemblMapping.txt", sep='\t')
            temp = pd.merge(left=ensemble, right=CPDB, how='right', left_on='Gene name', right_on='partner1').dropna()
            temp = temp.drop_duplicates().reset_index(drop=True)
            ppi = pd.merge(left=temp, right=ensemble, how='left', left_on='partner2', right_on='Gene name').dropna()
            ppi = ppi[['Protein stable ID_x', 'Protein stable ID_y']]
            ppi = ppi.drop_duplicates().reset_index(drop=True)
            print()


        unique_proteins = pd.DataFrame(set(ppi['protein1'].values) | set(ppi['protein2'])).sort_values(by=0)
        unique_proteins.reset_index(drop=True, inplace=True)
        ppi.reset_index(drop=True, inplace=True)
        num_interactions = ppi.shape[0]
        print("Get {} protein nodes, {} interactions.\n".format(len(unique_proteins), num_interactions))

        # Reorganize protein id
        unique_protein_id = pd.DataFrame(data={
            'ProteinID': unique_proteins[0].unique(),
            'MappedProteinID': pd.RangeIndex(unique_proteins.shape[0])
        })

        source = pd.merge(left=ppi['protein1'], right=unique_protein_id, how='left', left_on='protein1',
                          right_on='ProteinID')
        source = torch.from_numpy(source['MappedProteinID'].values)
        target = pd.merge(left=ppi['protein2'], right=unique_protein_id, how='left', left_on='protein2',
                          right_on='ProteinID')
        target = torch.from_numpy(target['MappedProteinID'].values)
        edge_index_ppi = torch.stack([source, target], dim=0)
        print("Finished to construct PPI !!!")

        # # Read Ensembl mapping
        # print("Reading Ensembl mapping, please wait ... \n")
        # ensembl_mapping = pd.read_table(os.path.join(self.file_path, "EnsemblMapping.txt"))
        # columns_keep = ['Protein stable ID', 'Gene name']
        # ensembl_mapping = ensembl_mapping[columns_keep]
        # ensembl_mapping = ensembl_mapping.dropna()
        # ensembl_mapping = ensembl_mapping.sort_values(by='Protein stable ID')
        # ensembl_mapping.reset_index(drop=True, inplace=True)
        #
        # ensembl_mapping = pd.merge(left=ensembl_mapping, right=unique_protein_id, how='left',
        #                            left_on='Protein stable ID', right_on='ProteinID')
        #
        # return edge_index_ppi, ensembl_mapping
        return edge_index_ppi, unique_protein_id


# if __name__ == '__main__':
#     file_path = "RawData\\Associations\\PPI"
#     constructor = PPIConstructor(file_path=file_path)
#     constructor.construct_ppi()
