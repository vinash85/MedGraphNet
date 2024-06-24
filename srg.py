import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc


class GCN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
                super(GCN, self).__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(GCNConv(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                x = self.convs[-1](x, edge_index)
                return x

class NodeEmbeddingNN(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(NodeEmbeddingNN, self).__init__()
                self.fc = torch.nn.Linear(in_channels, out_channels)

            def forward(self, x):
                return self.fc(x)


def train(train_data, train_edge_index, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    
    pos_samples = train_edge_index
    labels = train_data.y.to(device)
    out_pos = torch.sum(out[pos_samples[0]] * out[pos_samples[1]], dim=1)
    
    loss = criterion(out_pos, labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss.item()


def evaluate(data, edge_index, model, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data)
        
        pos_samples = edge_index
        labels = data.y.to(device)
        
        out_pos = torch.sum(out[pos_samples[0]] * out[pos_samples[1]], dim=1)
        
        loss = criterion(out_pos, labels)
        
        pred_labels = (out_pos > 0.5).float()
        
        acc = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
        roc_auc = roc_auc_score(labels.cpu().numpy(), out_pos.cpu().numpy())
        precision, recall, _ = precision_recall_curve(labels.cpu().numpy(), out_pos.cpu().numpy())
        aupr = auc(recall, precision)

    return loss.item(), acc, f1, roc_auc, aupr



def prepare_data(train_df, test_df, embeddings, entity1, entity2):
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    all_entities = np.unique(np.concatenate((train_df[entity1].unique(), test_df[entity1].unique())))
    all_genes = np.unique(np.concatenate((train_df[entity2].unique(), test_df[entity2].unique())))

    entity2idx = {entity: idx for idx, entity in enumerate(all_entities)}
    gene2idx = {gene: idx + len(all_entities) for idx, gene in enumerate(all_genes)}  

    train_edges = []
    train_labels = []
    for _, row in train_df.iterrows():
        entity_idx = entity2idx[row[entity1]]
        gene_idx = gene2idx[row[entity2]]
        train_edges.append([entity_idx, gene_idx])
        train_labels.append(row['Association'])

    # Split train edges and labels to train and val
    train_edges, val_edges, train_labels, val_labels = train_test_split(train_edges, train_labels, test_size=0.1, random_state=42)
    train_edges = np.array(train_edges)
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    train_labels = torch.tensor(train_labels, dtype=torch.float)

    val_edges = np.array(val_edges)
    val_edge_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous()
    val_labels = torch.tensor(val_labels, dtype=torch.float)

    test_edges = []
    test_labels = []
    for _, row in test_df.iterrows():
        entity_idx = entity2idx[row[entity1]]
        gene_idx = gene2idx[row[entity2]]
        test_edges.append([entity_idx, gene_idx])
        test_labels.append(row['Association'])

    test_edges = np.array(test_edges)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    num_nodes = len(all_entities) + len(all_genes)
    embedding_dim = 64
    node_embeddings = np.random.randn(num_nodes, embedding_dim).astype(np.float32)

    # for entity, idx in entity2idx.items():
    #     if entity in embeddings:
    #         node_embeddings[idx] = embeddings[entity]
    # for gene, idx in gene2idx.items():
    #     if gene in gene_embeddings:
    #         node_embeddings[idx] = gene_embeddings[gene]

    return train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings





# Load precomputed embeddings
df_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/gene_embeddings.csv')
df_disease = pd.read_csv('/data/sahulab/macaulay/Gene_disease/all_disease_embeddings.csv')
df_drug = pd.read_csv('/data/sahulab/macaulay/Gene_disease/all_drug_embeddings.csv')
df_phenotype = pd.read_csv('/data/sahulab/macaulay/Gene_disease/all_phenotype_embeddings.csv')

# training datasets
df_disease_gene_train_all_disease = pd.read_csv('data/Disease_Gene_association_train_Disease_all.csv')
df_disease_gene_train_all_gene = pd.read_csv('data/Disease_Gene_association_train_Gene_all.csv')
df_disease_gene_train_one_disease = pd.read_csv('data/Disease_Gene_association_train_Disease_one.csv')
df_disease_gene_train_one_gene = pd.read_csv('data/Disease_Gene_association_train_Gene_one.csv')

df_disease_drug_train_all_disease = pd.read_csv('data/Disease_Drug_association_train_Disease_all.csv')
df_disease_drug_train_all_drug = pd.read_csv('data/Disease_Drug_association_train_Drug_all.csv')
df_disease_drug_train_one_disease = pd.read_csv('data/Disease_Drug_association_train_Disease_one.csv')
df_disease_drug_train_one_drug = pd.read_csv('data/Disease_Drug_association_train_Drug_one.csv')

df_disease_phenotype_train_all_disease = pd.read_csv('data/Disease_Phenotype_association_train_Disease_all.csv')
df_disease_phenotype_train_all_phenotype = pd.read_csv('data/Disease_Phenotype_association_train_Phenotype_all.csv')
df_disease_phenotype_train_one_disease = pd.read_csv('data/Disease_Phenotype_association_train_Disease_one.csv')
df_disease_phenotype_train_one_phenotype = pd.read_csv('data/Disease_Phenotype_association_train_Phenotype_one.csv')

df_drug_phenotype_train_all_drug = pd.read_csv('data/Drug_Phenotype_association_train_Drug_all.csv')
df_drug_phenotype_train_all_phenotype = pd.read_csv('data/Drug_Phenotype_association_train_Phenotype_all.csv')
df_drug_phenotype_train_one_drug = pd.read_csv('data/Drug_Phenotype_association_train_Drug_one.csv')
df_drug_phenotype_train_one_phenotype = pd.read_csv('data/Drug_Phenotype_association_train_Phenotype_one.csv')

df_gene_drug_train_all_drug = pd.read_csv('data/Gene_Drug_association_train_Drug_all.csv')
df_gene_drug_train_all_gene = pd.read_csv('data/Gene_Drug_association_train_Gene_all.csv')
df_gene_drug_train_one_drug = pd.read_csv('data/Gene_Drug_association_train_Drug_one.csv')
df_gene_drug_train_one_gene = pd.read_csv('data/Gene_Drug_association_train_Gene_one.csv')

df_phenotype_gene_train_phenotype = pd.read_csv('data/Phenotype_Gene_association_train_Phenotype_all.csv')
df_phenotype_gene_train_gene = pd.read_csv('data/Phenotype_Gene_association_train_Gene_all.csv')
df_phenotype_gene_train_one_phenotype = pd.read_csv('data/Phenotype_Gene_association_train_Phenotype_one.csv')
df_phenotype_gene_train_one_gene = pd.read_csv('data/Phenotype_Gene_association_train_Gene_one.csv')

#test datasets
df_disease_gene_test_all_disease = pd.read_csv('data/Disease_Gene_association_test_Disease_all.csv')
df_disease_gene_test_all_gene = pd.read_csv('data/Disease_Gene_association_test_Gene_all.csv')
df_disease_gene_test_one_disease = pd.read_csv('data/Disease_Gene_association_test_Disease_one.csv')
df_disease_gene_test_one_gene = pd.read_csv('data/Disease_Gene_association_test_Gene_one.csv')

df_disease_drug_test_all_disease = pd.read_csv('data/Disease_Drug_association_test_Disease_all.csv')
df_disease_drug_test_all_drug = pd.read_csv('data/Disease_Drug_association_test_Drug_all.csv')
df_disease_drug_test_one_disease = pd.read_csv('data/Disease_Drug_association_test_Disease_one.csv')
df_disease_drug_test_one_drug = pd.read_csv('data/Disease_Drug_association_test_Drug_one.csv')

df_disease_phenotype_test_all_disease = pd.read_csv('data/Disease_Phenotype_association_test_Disease_all.csv')
df_disease_phenotype_test_all_phenotype = pd.read_csv('data/Disease_Phenotype_association_test_Phenotype_all.csv')
df_disease_phenotype_test_one_disease = pd.read_csv('data/Disease_Phenotype_association_test_Disease_one.csv')
df_disease_phenotype_test_one_phenotype = pd.read_csv('data/Disease_Phenotype_association_test_Phenotype_one.csv')

df_drug_phenotype_test_all_drug = pd.read_csv('data/Drug_Phenotype_association_test_Drug_all.csv')
df_drug_phenotype_test_all_phenotype = pd.read_csv('data/Drug_Phenotype_association_test_Phenotype_all.csv')
df_drug_phenotype_test_one_drug = pd.read_csv('data/Drug_Phenotype_association_test_Drug_one.csv')
df_drug_phenotype_test_one_phenotype = pd.read_csv('data/Drug_Phenotype_association_test_Phenotype_one.csv')

df_gene_drug_test_all_drug = pd.read_csv('data/Gene_Drug_association_test_Drug_all.csv')
df_gene_drug_test_all_gene = pd.read_csv('data/Gene_Drug_association_test_Gene_all.csv')
df_gene_drug_test_one_drug = pd.read_csv('data/Gene_Drug_association_test_Drug_one.csv')
df_gene_drug_test_one_gene = pd.read_csv('data/Gene_Drug_association_test_Gene_one.csv')

df_phenotype_gene_test_phenotype = pd.read_csv('data/Phenotype_Gene_association_test_Phenotype_all.csv')
df_phenotype_gene_test_gene = pd.read_csv('data/Phenotype_Gene_association_test_Gene_all.csv')
df_phenotype_gene_test_one_phenotype = pd.read_csv('data/Phenotype_Gene_association_test_Phenotype_one.csv')
df_phenotype_gene_test_one_gene = pd.read_csv('data/Phenotype_Gene_association_test_Gene_one.csv')






gene_embeddings = {row['Gene name']: np.array(row[1:], dtype=np.float32) for index, row in df_gene.iterrows()}
disease_embeddings = {row['Disease']: np.array(row[1:], dtype=np.float32) for index, row in df_disease.iterrows()}
drug_embeddings = {row['Drug']: np.array(row[1:], dtype=np.float32) for index, row in df_drug.iterrows()}
phenotype_embeddings = {row['Phenotype']: np.array(row[1:], dtype=np.float32) for index, row in df_phenotype.iterrows()}


train_df_list = [df_disease_gene_train_all_disease, df_disease_gene_train_all_gene, df_disease_gene_train_one_disease, df_disease_gene_train_one_gene, df_disease_drug_train_all_disease, df_disease_drug_train_all_drug, df_disease_drug_train_one_disease, df_disease_drug_train_one_drug, df_disease_phenotype_train_all_disease, df_disease_phenotype_train_all_phenotype, df_disease_phenotype_train_one_disease, df_disease_phenotype_train_one_phenotype,
                  df_drug_phenotype_train_all_drug, df_drug_phenotype_train_all_phenotype, df_drug_phenotype_train_one_drug, df_drug_phenotype_train_one_phenotype,
                    df_gene_drug_train_all_drug, df_gene_drug_train_all_gene, df_gene_drug_train_one_drug, df_gene_drug_train_one_gene,
                    df_phenotype_gene_train_phenotype, df_phenotype_gene_train_gene, df_phenotype_gene_train_one_phenotype, df_phenotype_gene_train_one_gene
                 ]
test_df_list = [df_disease_gene_test_all_disease, df_disease_gene_test_all_gene, df_disease_gene_test_one_disease, df_disease_gene_test_one_gene, df_disease_drug_test_all_disease, df_disease_drug_test_all_drug, df_disease_drug_test_one_disease, df_disease_drug_test_one_drug, df_disease_phenotype_test_all_disease, df_disease_phenotype_test_all_phenotype, df_disease_phenotype_test_one_disease, df_disease_phenotype_test_one_phenotype, 
                df_drug_phenotype_test_all_drug, df_drug_phenotype_test_all_phenotype, df_drug_phenotype_test_one_drug, df_drug_phenotype_test_one_phenotype,
                  df_gene_drug_test_all_drug, df_gene_drug_test_all_gene, df_gene_drug_test_one_drug, df_gene_drug_test_one_gene, 
                  df_phenotype_gene_test_phenotype, df_phenotype_gene_test_gene, df_phenotype_gene_test_one_phenotype, df_phenotype_gene_test_one_gene
                ]
category_names = ['Disease_gene_all_disease', 'Disease_gene_all_gene', 'Disease_gene_one_disease', 'Disease_gene_one_gene', 'Disease_drug_all_disease', 'Disease_drug_all_drug', 'Disease_drug_one_disease', 'Disease_drug_one_drug', 'Disease_phenotype_all_disease', 'Disease_phenotype_all_phenotype', 'Disease_phenotype_one_disease', 'Disease_phenotype_one_phenotype', 'Drug_phenotype_all_drug', 'Drug_phenotype_all_phenotype', 'Drug_phenotype_one_drug', 'Drug_phenotype_one_phenotype', 'Gene_drug_all_drug', 'Gene_drug_all_gene', 'Gene_drug_one_drug', 'Gene_drug_one_gene',
                  'Phenotype_gene_all_phenotype', 'Phenotype_gene_all_gene', 'Phenotype_gene_one_phenotype', 'Phenotype_gene_one_gene'
                  ]

for train_df, test_df, category_name in zip(train_df_list, test_df_list, category_names):
    columns = train_df.columns
    if 'Disease' in columns and 'Gene' in columns and 'Association' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, gene_embeddings, 'Disease', 'Gene')

    elif 'Phenotype' in columns and 'Gene' in columns and 'Association' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, gene_embeddings, 'Phenotype', 'Gene')

    elif 'Drug' in columns and 'Gene' in columns and 'Association' in columns:    
         
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, gene_embeddings, 'Drug', 'Gene')

    elif 'Disease' in columns and 'Drug' in columns and 'Association' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, drug_embeddings, 'Disease', 'Drug')

    elif 'Disease' in columns and 'Phenotype' in columns and 'Association' in columns:
         
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, phenotype_embeddings, 'Disease', 'Phenotype')

    elif 'Drug' in columns and 'Phenotype' in columns and 'Association' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings = prepare_data(train_df, test_df, phenotype_embeddings, 'Drug', 'Phenotype')




    # embedding_dim = 768
    output_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # node_embedding_nn = NodeEmbeddingNN(embedding_dim, output_dim).to(device)
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float).to(device)
    # x = node_embedding_nn(node_embeddings)
    x = node_embeddings

    
        
    num_layers = 4
    model = GCN(in_channels=output_dim, hidden_channels=32, out_channels=64, num_layers=num_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = Data(x=x, edge_index=train_edge_index, y=train_labels).to(device)
    test_data = Data(x=x, edge_index=test_edge_index, y=test_labels).to(device)
    val_data = Data(x=x, edge_index=val_edge_index, y=val_labels).to(device)

    for epoch in range(1, 300):
        loss = train(train_data, train_edge_index, model, optimizer, criterion)
        val_loss, val_acc, val_f1, val_roc_auc, val_aupr = evaluate(val_data, val_edge_index, model, criterion)
        print(category_name)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val ROC AUC: {val_roc_auc:.4f}, Val AUPR: {val_aupr:.4f}')
        
    
    test_loss, test_acc, test_f1, test_roc_auc, test_aupr = evaluate(test_data, test_edge_index, model, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}')
    with open('single_graph_results.txt', 'a') as f:
        f.write(f'{category_name}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}\n')




