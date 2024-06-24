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

def print_dict_head(d, n=5):
    """Print the first n items of a dictionary."""
    for i, (key, value) in enumerate(d.items()):
        if i >= n:
            break
        print(f"{key}")

def generate_negative_samples(edge_index, num_nodes, num_neg_samples):
    edge_set = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1)))
    negative_samples = []
    while len(negative_samples) < num_neg_samples:
        i, j = np.random.randint(0, num_nodes, size=2)
        if (i, j) not in edge_set and (j, i) not in edge_set:
            negative_samples.append([i, j])
    return torch.tensor(negative_samples, dtype=torch.long).t().contiguous().to(device)



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


# def train(train_data, train_edge_index, model, optimizer, criterion):
#     model.train()
#     optimizer.zero_grad()
#     out = model(train_data)
    
#     pos_samples = train_edge_index
#     labels = train_data.y.to(device)
    
#     # Calculate the similarity scores for edges
#     out_pos = torch.sum(out[pos_samples[0]] * out[pos_samples[1]], dim=1)
    
#     loss = criterion(out_pos, labels)
#     loss.backward(retain_graph=True)
#     optimizer.step()

#     return loss.item()


# Training the model
def train(train_data, train_edge_index, model, optimizer, criterion, num_nodes, device):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    
    pos_samples = train_edge_index.to(device)
    neg_samples = generate_negative_samples(train_edge_index, num_nodes, pos_samples.size(1))
    labels = torch.cat([torch.ones(pos_samples.size(1)), torch.zeros(neg_samples.size(1))]).to(device)
    
    samples = torch.cat([pos_samples, neg_samples], dim=1)
    
    out_pos = torch.sum(out[samples[0, :pos_samples.size(1)]] * out[samples[1, :pos_samples.size(1)]], dim=1)
    out_neg = torch.sum(out[samples[0, pos_samples.size(1):]] * out[samples[1, pos_samples.size(1):]], dim=1)
    out = torch.cat([out_pos, out_neg])
    
    loss = criterion(out, labels)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()



# Evaluate function for a given test data
def evaluate_test_data(test_edge_index, model, criterion, x, num_nodes, device):
    test_data = Data(x=x, edge_index=test_edge_index.to(device))
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        
        pos_samples = test_edge_index.to(device)
        neg_samples = generate_negative_samples(test_edge_index, num_nodes, pos_samples.size(1))
        labels = torch.cat([torch.ones(pos_samples.size(1)), torch.zeros(neg_samples.size(1))]).to(device)
        
        samples = torch.cat([pos_samples, neg_samples], dim=1)
        
        out_pos = torch.sum(out[samples[0, :pos_samples.size(1)]] * out[samples[1, :pos_samples.size(1)]], dim=1)
        out_neg = torch.sum(out[samples[0, pos_samples.size(1):]] * out[samples[1, pos_samples.size(1):]], dim=1)
        out = torch.cat([out_pos, out_neg])
        
        probs = torch.sigmoid(out).cpu().numpy()
        labels = labels.cpu().numpy()
        
        accuracy = accuracy_score(labels, probs >= 0.5)
        precision = precision_score(labels, probs >= 0.5)
        recall = recall_score(labels, probs >= 0.5)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, probs >= 0.5)
        cm = confusion_matrix(labels, probs >= 0.5)
        # precision, recall, _ = precision_recall_curve(labels, probs)
        # aupr = auc(recall, precision)
        
        print(f'Accuracy: {accuracy:.4f}')
        # print(f'Precision: {precision:.4f}')
        # print(f'Recall: {recall:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:')
        print(f'True Negative: {cm[0, 0]}, False Positive: {cm[0, 1]}')
        print(f'False Negative: {cm[1, 0]}, True Positive: {cm[1, 1]}')
        
        loss = criterion(out, torch.tensor(labels, dtype=torch.float).to(device))
        return loss.item(), accuracy, f1, auc, aupr


# def evaluate(data, edge_index, model, criterion):
#     model.eval()
#     with torch.no_grad():
#         out = model(data)
        
#         pos_samples = edge_index
#         labels = data.y.to(device)
        
#         # Calculate the similarity scores for edges
#         out_pos = torch.sum(out[pos_samples[0]] * out[pos_samples[1]], dim=1)
        
#         loss = criterion(out_pos, labels)
        
#         # Convert predictions to binary
#         pred_labels = (out_pos > 0.5).float()
        
#         # Calculate metrics
#         acc = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
#         f1 = f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
#         roc_auc = roc_auc_score(labels.cpu().numpy(), out_pos.cpu().numpy())
#         precision, recall, _ = precision_recall_curve(labels.cpu().numpy(), out_pos.cpu().numpy())
#         aupr = auc(recall, precision)

#     return loss.item(), acc, f1, roc_auc, aupr

def validate(val_data, val_edge_index, model, criterion, num_nodes, device):
    model.eval()
    with torch.no_grad():
        out = model(val_data)
        
        pos_samples = val_edge_index.to(device)
        neg_samples = generate_negative_samples(val_edge_index, num_nodes, pos_samples.size(1))
        labels = torch.cat([torch.ones(pos_samples.size(1)), torch.zeros(neg_samples.size(1))]).to(device)
        
        samples = torch.cat([pos_samples, neg_samples], dim=1)
        
        out_pos = torch.sum(out[samples[0, :pos_samples.size(1)]] * out[samples[1, :pos_samples.size(1)]], dim=1)
        out_neg = torch.sum(out[samples[0, pos_samples.size(1):]] * out[samples[1, pos_samples.size(1):]], dim=1)
        out = torch.cat([out_pos, out_neg])
        
        probs = torch.sigmoid(out).cpu().numpy()
        labels = labels.cpu().numpy()
        
        accuracy = accuracy_score(labels, probs >= 0.5)
        precision = precision_score(labels, probs >= 0.5)
        recall = recall_score(labels, probs >= 0.5)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, probs >= 0.5)
        precision, recall, _ = precision_recall_curve(labels, probs)
        # aupr = auc(recall, precision)
        
        loss = criterion(out, torch.tensor(labels, dtype=torch.float).to(device))
        return loss.item(), accuracy,f1, auc, aupr 

def disease_gene_edges(df_disease_drug, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_disease_phenotype[entity1], df_disease_drug[entity1], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_phenotype_gene[entity2], df_gene_drug[entity2], df_gene_gene['Gene1'], df_gene_gene['Gene2'], train_df[entity2], test_df[entity2]]))
    all_phenotype = np.unique(np.concatenate([df_disease_phenotype['Phenotype'], df_phenotype_gene['Phenotype'], df_drug_phenotype['Phenotype']]))
    all_drugs = np.unique(np.concatenate([df_disease_drug['Drug'], df_gene_drug['Drug'], df_drug_phenotype['Drug']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_phenotype) + len(all_drugs)

    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)}  
    phenotype_idx = {phenotype: idx + len(all_entities1) + len(all_entities2) for idx, phenotype in enumerate(all_phenotype)}
    drug_idx = {drug: idx + len(all_entities1) + len(all_entities2) + len(all_phenotype) for idx, drug in enumerate(all_drugs)}
    train_edges = []
    train_labels = []

    # Disease-Gene edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    #phenotype_gene edges
    for _, row in df_phenotype_gene.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        entity2_to_idx = entity2_idx[row['Gene']]
        train_edges.append([phenotype_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Drug edges
    for _, row in df_disease_drug.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity1_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])
        

    # Disease-Phenotype edges
    for _, row in df_disease_phenotype.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        train_edges.append([entity1_to_idx, phenotype_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Drug edges
    for _, row in df_gene_drug.iterrows():
        entity2_to_idx = entity2_idx[row['Gene']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity2_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges
    for _, row in df_gene_gene.iterrows():
        entity2_1_to_idx = entity2_idx[row['Gene1']]
        entity2_2_to_idx = entity2_idx[row['Gene2']]
        train_edges.append([entity2_1_to_idx, entity2_2_to_idx])
        # train_labels.append(row['Association'])

    # phenotype_drug edges
    for _, row in df_drug_phenotype.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([phenotype_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    print('All edges created')

    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)
    print_dict_head(disease_embeddings, 5)  
    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in phenotype_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]
    for entity, idx in drug_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]







    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings


def phenotype_gene_edges(df_disease_drug, df_disease_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_disease_phenotype[entity1], df_drug_phenotype[entity1], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_disease_gene[entity2], df_gene_drug[entity2], df_gene_gene['Gene1'], df_gene_gene['Gene2'], train_df[entity2], test_df[entity2]]))
    all_disease = np.unique(np.concatenate([df_disease_gene['Disease'], df_disease_drug['Disease'], df_disease_phenotype['Disease']]))
    all_drugs = np.unique(np.concatenate([df_drug_phenotype['Drug'], df_gene_drug['Drug'], df_disease_drug['Drug']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_disease) + len(all_drugs)


    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)} 
    disease_idx = {disease: idx + len(all_entities1) + len(all_entities2) for idx, disease in enumerate(all_disease)}
    drug_idx = {drug: idx + len(all_entities1) + len(all_entities2) + len(all_disease) for idx, drug in enumerate(all_drugs)}


    train_edges = []
    train_labels = []

    # Phenotype-Gene edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Gene edges
    for _, row in df_disease_gene.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity2_to_idx = entity2_idx[row['Gene']]
        train_edges.append([disease_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Drug edges
    for _, row in df_disease_drug.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([disease_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Phenotype edges
    for _, row in df_disease_phenotype.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity1_to_idx = entity1_idx[row['Phenotype']]
        train_edges.append([disease_to_idx, entity1_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Drug edges
    for _, row in df_gene_drug.iterrows():
        entity2_to_idx = entity2_idx[row['Gene']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity2_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges
    for _, row in df_gene_gene.iterrows():
        entity2_1_to_idx = entity2_idx[row['Gene1']]
        entity2_2_to_idx = entity2_idx[row['Gene2']]
        train_edges.append([entity2_1_to_idx, entity2_2_to_idx])
        # train_labels.append(row['Association'])

    # phenotype_drug edges
    for _, row in df_drug_phenotype.iterrows():
        entity1_to_idx = entity1_idx[row['Phenotype']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity1_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    print('All edges created')


    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)

    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in disease_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]
    for entity, idx in drug_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]


    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings




def disease_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_disease_gene[entity1], df_disease_phenotype[entity1], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_gene_drug[entity2], df_drug_phenotype[entity2], train_df[entity2], test_df[entity2]]))
    all_genes = np.unique(np.concatenate([df_disease_gene['Gene'], df_phenotype_gene['Gene'], df_gene_drug['Gene'], df_gene_gene['Gene1'], df_gene_gene['Gene2']]))
    all_phenotype = np.unique(np.concatenate([df_disease_phenotype['Phenotype'], df_phenotype_gene['Phenotype'], df_drug_phenotype['Phenotype']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_genes) + len(all_phenotype)

    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)}  # 
    gene_idx = {gene: idx + len(all_entities1) + len(all_entities2) for idx, gene in enumerate(all_genes)}
    phenotype_idx = {phenotype: idx + len(all_entities1) + len(all_entities2) + len(all_genes) for idx, phenotype in enumerate(all_phenotype)}


    train_edges = []
    train_labels = []

    # Disease-Drug edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Gene edges
    for _, row in df_disease_gene.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([entity1_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])
        

    #phenotype_gene edges
    for _, row in df_phenotype_gene.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([phenotype_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Phenotype edges
    for _, row in df_disease_phenotype.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        train_edges.append([entity1_to_idx, phenotype_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Drug edges
    for _, row in df_gene_drug.iterrows():
        gene_to_idx = gene_idx[row['Gene']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([gene_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges
    for _, row in df_gene_gene.iterrows():
        gene1_to_idx = gene_idx[row['Gene1']]
        gene2_to_idx = gene_idx[row['Gene2']]
        train_edges.append([gene1_to_idx, gene2_to_idx])
        # train_labels.append(row['Association'])

    # phenotype_drug edges
    for _, row in df_drug_phenotype.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([phenotype_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    print('All edges created')

    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)

    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]
    for entity, idx in gene_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in phenotype_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]

    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings


def disease_phenotype_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_disease_gene[entity1], df_disease_drug[entity1], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_phenotype_gene[entity2], df_drug_phenotype[entity2], train_df[entity2], test_df[entity2]]))
    all_genes = np.unique(np.concatenate([df_disease_gene['Gene'], df_phenotype_gene['Gene'], df_gene_drug['Gene'], df_gene_gene['Gene1'], df_gene_gene['Gene2']]))
    all_drugs = np.unique(np.concatenate([df_disease_drug['Drug'], df_gene_drug['Drug'], df_drug_phenotype['Drug']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_genes) + len(all_drugs)

    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)}  
    gene_idx = {gene: idx + len(all_entities1) + len(all_entities2) for idx, gene in enumerate(all_genes)}
    drug_idx = {drug: idx + len(all_entities1) + len(all_entities2) + len(all_genes) for idx, drug in enumerate(all_drugs)}


    train_edges = []
    train_labels = []

    # Disease-Phenotype edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Gene edges
    for _, row in df_disease_gene.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([entity1_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])
        

    #phenotype_gene edges
    for _, row in df_phenotype_gene.iterrows():
        entity2_to_idx = entity2_idx[row['Phenotype']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([entity2_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Drug edges
    for _, row in df_disease_drug.iterrows():
        entity1_to_idx = entity1_idx[row['Disease']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity1_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Drug edges
    for _, row in df_gene_drug.iterrows():
        gene_to_idx = gene_idx[row['Gene']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([gene_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges

    for _, row in df_gene_gene.iterrows():
        gene1_to_idx = gene_idx[row['Gene1']]
        gene2_to_idx = gene_idx[row['Gene2']]
        train_edges.append([gene1_to_idx, gene2_to_idx])
        # train_labels.append(row['Association'])

    # phenotype_drug edges
    for _, row in df_drug_phenotype.iterrows():
        entity2_to_idx = entity2_idx[row['Phenotype']]
        drug_to_idx = drug_idx[row['Drug']]
        train_edges.append([entity2_to_idx, drug_to_idx])
        # train_labels.append(row['Association'])

    print('All edges created')

    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)

    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]
    for entity, idx in gene_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in drug_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]


    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings


def gene_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_disease_phenotype, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_disease_gene[entity1], df_phenotype_gene[entity1], df_gene_gene['Gene1'], df_gene_gene['Gene2'], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_disease_drug[entity2], df_drug_phenotype[entity2], train_df[entity2], test_df[entity2]]))
    all_disease = np.unique(np.concatenate([df_disease_gene['Disease'], df_disease_drug['Disease'], df_disease_phenotype['Disease']]))
    all_phenotype = np.unique(np.concatenate([df_disease_phenotype['Phenotype'], df_phenotype_gene['Phenotype'], df_drug_phenotype['Phenotype']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_disease) + len(all_phenotype)

    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)}  
    disease_idx = {disease: idx + len(all_entities1) + len(all_entities2) for idx, disease in enumerate(all_disease)}
    phenotype_idx = {phenotype: idx + len(all_entities1) + len(all_entities2) + len(all_disease) for idx, phenotype in enumerate(all_phenotype)}


    train_edges = []
    train_labels = []

    # Gene-Drug edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Gene edges
    for _, row in df_disease_gene.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity1_to_idx = entity1_idx[row['Gene']]
        train_edges.append([disease_to_idx, entity1_to_idx])
        # train_labels.append(row['Association'])
        

    #phenotype_gene edges
    for _, row in df_phenotype_gene.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        entity1_to_idx = entity1_idx[row['Gene']]
        train_edges.append([phenotype_to_idx, entity1_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Drug edges
    for _, row in df_disease_drug.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([disease_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Phenotype edges
    for _, row in df_disease_phenotype.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        train_edges.append([disease_to_idx, phenotype_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges
    for _, row in df_gene_gene.iterrows():
        entity1_1_to_idx = entity1_idx[row['Gene1']]
        entity1_2_to_idx = entity1_idx[row['Gene2']]
        train_edges.append([entity1_1_to_idx, entity1_2_to_idx])
        # train_labels.append(row['Association'])

    # phenotype_drug edges
    for _, row in df_drug_phenotype.iterrows():
        phenotype_to_idx = phenotype_idx[row['Phenotype']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([phenotype_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    print('All edges created')

    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)

    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]
    for entity, idx in disease_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]
    for entity, idx in phenotype_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]


    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings


def phenotype_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_disease_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings):
    all_entities1 = np.unique(np.concatenate([df_phenotype_gene[entity1], df_disease_phenotype[entity1], train_df[entity1], test_df[entity1]]))
    all_entities2 = np.unique(np.concatenate([df_gene_drug[entity2], df_disease_drug[entity2], train_df[entity2], test_df[entity2]]))
    all_genes = np.unique(np.concatenate([df_disease_gene['Gene'], df_phenotype_gene['Gene'], df_gene_drug['Gene'], df_gene_gene['Gene1'], df_gene_gene['Gene2']]))
    all_disease = np.unique(np.concatenate([df_disease_gene['Disease'], df_disease_drug['Disease'], df_disease_phenotype['Disease']]))
    node_enb_length = len(all_entities1) + len(all_entities2) + len(all_genes) + len(all_disease)

    entity1_idx = {entity1: idx for idx, entity1 in enumerate(all_entities1)}
    entity2_idx = {entity2: idx + len(all_entities1) for idx, entity2 in enumerate(all_entities2)}  
    gene_idx = {gene: idx + len(all_entities1) + len(all_entities2) for idx, gene in enumerate(all_genes)}
    disease_idx = {disease: idx + len(all_entities1) + len(all_entities2) + len(all_genes) for idx, disease in enumerate(all_disease)}

    train_edges = []
    train_labels = []

    # Phenotype-Drug edges
    for _, row in train_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        train_edges.append([entity1_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Gene edges
    for _, row in df_disease_gene.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([disease_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])

    #phenotype_gene edges
    for _, row in df_phenotype_gene.iterrows():
        entity1_to_idx = entity1_idx[row['Phenotype']]
        gene_to_idx = gene_idx[row['Gene']]
        train_edges.append([entity1_to_idx, gene_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Drug edges
    for _, row in df_disease_drug.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([disease_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Drug edges
    for _, row in df_gene_drug.iterrows():
        gene_to_idx = gene_idx[row['Gene']]
        entity2_to_idx = entity2_idx[row['Drug']]
        train_edges.append([gene_to_idx, entity2_to_idx])
        # train_labels.append(row['Association'])

    # Gene-Gene edges
    for _, row in df_gene_gene.iterrows():
        gene1_to_idx = gene_idx[row['Gene1']]
        gene2_to_idx = gene_idx[row['Gene2']]
        train_edges.append([gene1_to_idx, gene2_to_idx])
        # train_labels.append(row['Association'])

    # Disease-Phenotype edges
    for _, row in df_disease_phenotype.iterrows():
        disease_to_idx = disease_idx[row['Disease']]
        entity1_to_idx = entity1_idx[row['Phenotype']]
        train_edges.append([disease_to_idx, entity1_to_idx])
        # train_labels.append(row['Association'])


    print('All edges created')

    embedding_dim = 768
    node_embeddings = np.random.randn(node_enb_length, embedding_dim).astype(np.float32)

    for entity, idx in entity1_idx.items():
        node_embeddings[idx] = phenotype_embeddings[entity]
    for entity, idx in entity2_idx.items():
        node_embeddings[idx] = drug_embeddings[entity]
    for entity, idx in gene_idx.items():
        node_embeddings[idx] = gene_embeddings[entity]
    for entity, idx in disease_idx.items():
        node_embeddings[idx] = disease_embeddings[entity]


    return train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings



def prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, entity1, entity2):



    # Load all datasets
    df_disease_drug = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_drug_association_f.csv')
    df_phenotype_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/phenotype_gene_association_f.csv')
    df_disease_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_gene_association_f.csv')
    df_disease_phenotype = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_phenotype_association_f.csv')
    df_gene_drug = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/gene_drug_association_f.csv')
    df_gene_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/gene_gene_association_f.csv')
    df_drug_phenotype = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/drug_phenotype_association_f.csv')

    df_disease_drug = df_disease_drug[df_disease_drug['Association'] != 0]
    df_phenotype_gene = df_phenotype_gene[df_phenotype_gene['Association'] != 0]
    df_disease_gene = df_disease_gene[df_disease_gene['Association'] != 0]
    df_disease_phenotype = df_disease_phenotype[df_disease_phenotype['Association'] != 0]
    df_gene_drug = df_gene_drug[df_gene_drug['Association'] != 0]
    df_gene_gene = df_gene_gene[df_gene_gene['Association'] != 0]
    df_drug_phenotype = df_drug_phenotype[df_drug_phenotype['Association'] != 0]

    # shuffle the dataframes
    df_disease_drug = df_disease_drug.sample(frac=1, random_state=42).reset_index(drop=True)
    df_phenotype_gene = df_phenotype_gene.sample(frac=1, random_state=42).reset_index(drop=True)
    df_disease_gene = df_disease_gene.sample(frac=1, random_state=42).reset_index(drop=True)
    df_disease_phenotype = df_disease_phenotype.sample(frac=1, random_state=42).reset_index(drop=True)
    df_gene_drug = df_gene_drug.sample(frac=1, random_state=42).reset_index(drop=True)
    df_gene_gene = df_gene_gene.sample(frac=1, random_state=42).reset_index(drop=True)
    df_drug_phenotype = df_drug_phenotype.sample(frac=1, random_state=42).reset_index(drop=True)

    # diseases = np.unique(np.concatenate([df_disease_drug['Disease'], df_disease_gene['Disease'], df_disease_phenotype['Disease']]))
    # drugs = np.unique(np.concatenate([df_disease_drug['Drug'], df_gene_drug['Drug'], df_drug_phenotype['Drug']]))
    # phenotypes = np.unique(np.concatenate([df_disease_phenotype['Phenotype'], df_phenotype_gene['Phenotype'], df_drug_phenotype['Phenotype']]))
    # genes = np.unique(np.concatenate([df_disease_gene['Gene'], df_phenotype_gene['Gene'], df_gene_drug['Gene'], df_gene_gene['Gene1'], df_gene_gene['Gene2']]))
    
    
    train_df = train_df[train_df['Association'] != 0]
    test_df = test_df[test_df['Association'] != 0]
    # Shuffle the dataframes
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    


    columns = train_df.columns
    if 'Disease' in columns and 'Gene' in columns:
        if entity1 == 'Disease' and entity2 == 'Gene':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_gene_edges(df_disease_drug, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)

        elif entity1 == 'Gene' and entity2 == 'Disease':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_gene_edges(df_disease_drug, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
    
    elif 'Phenotype' in columns and 'Gene' in columns:  
        if entity1 == 'Phenotype' and entity2 == 'Gene':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = phenotype_gene_edges(df_disease_drug, df_disease_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
        elif entity1 == 'Gene' and entity2 == 'Phenotype':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = phenotype_gene_edges(df_disease_drug, df_disease_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
            
    elif 'Disease' in columns and 'Drug' in columns:
        if entity1 == 'Disease' and entity2 == 'Drug':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
        elif entity1 == 'Drug' and entity2 == 'Disease':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_phenotype, df_gene_drug, df_gene_gene, df_drug_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)

    elif 'Disease' in columns and 'Phenotype' in columns:
        if entity1 == 'Disease' and entity2 == 'Phenotype':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_phenotype_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
        elif entity1 == 'Phenotype' and entity2 == 'Disease':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = disease_phenotype_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_drug_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)

    elif 'Gene' in columns and 'Drug' in columns:
        if entity1 == 'Gene' and entity2 == 'Drug':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = gene_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_disease_phenotype, df_gene_gene, df_drug_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
        elif entity1 == 'Drug' and entity2 == 'Gene':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = gene_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_disease_phenotype, df_gene_gene, df_drug_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)

    elif 'Phenotype' in columns and 'Drug' in columns:
        if entity1 == 'Phenotype' and entity2 == 'Drug':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = phenotype_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_disease_phenotype, entity1, entity2, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)
        elif entity1 == 'Drug' and entity2 == 'Phenotype':
            train_edges, train_labels, entity1_idx, entity2_idx, node_enb_length, node_embeddings = phenotype_drug_edges(df_disease_gene, df_phenotype_gene, df_disease_drug, df_gene_drug, df_gene_gene, df_disease_phenotype, entity2, entity1, train_df, test_df, disease_embeddings, gene_embeddings, phenotype_embeddings, drug_embeddings)

    else:
        print('Invalid entity1 and entity2 combination')
        return                       

    # Split train edges and labels to train and val
    train_edges, val_edges= train_test_split(train_edges, test_size=0.1, random_state=42)
    train_edges = np.array(train_edges)
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    # train_labels = torch.tensor(train_labels, dtype=torch.float)

    val_edges = np.array(val_edges)
    val_edge_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous()
    val_labels = 1#torch.tensor(val_labels, dtype=torch.float)

    # Build test edges and labels
    test_edges = []
    # test_labels = []
    for _, row in test_df.iterrows():
        entity1_to_idx = entity1_idx[row[entity1]]
        entity2_to_idx = entity2_idx[row[entity2]]
        test_edges.append([entity1_to_idx, entity2_to_idx])
        # test_labels.append(row['Association'])

    test_edges = np.array(test_edges)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    test_labels = 1#torch.tensor(test_labels, dtype=torch.float)

    
    return train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length


# # Load all datasets
# df_disease_drug = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_drug_association_f.csv')
# df_phenotype_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/phenotype_gene_association_f.csv')
# df_disease_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_gene_association_f.csv')
# df_disease_phenotype = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/disease_phenotype_association_f.csv')
# df_gene_drug = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/gene_drug_association_f.csv')
# df_gene_gene = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/gene_gene_association_f.csv')
# df_drug_phenotype = pd.read_csv('/data/sahulab/macaulay/Gene_disease/unique_tests/datas/drug_phenotype_association_f.csv')


# # shuffle the dataframes
# df_disease_drug = df_disease_drug.sample(frac=1, random_state=42).reset_index(drop=True)
# df_phenotype_gene = df_phenotype_gene.sample(frac=1, random_state=42).reset_index(drop=True)
# df_disease_gene = df_disease_gene.sample(frac=1, random_state=42).reset_index(drop=True)
# df_disease_phenotype = df_disease_phenotype.sample(frac=1, random_state=42).reset_index(drop=True)
# df_gene_drug = df_gene_drug.sample(frac=1, random_state=42).reset_index(drop=True)
# df_gene_gene = df_gene_gene.sample(frac=1, random_state=42).reset_index(drop=True)
# df_drug_phenotype = df_drug_phenotype.sample(frac=1, random_state=42).reset_index(drop=True)

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
    if 'Disease' in columns and 'Gene' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Disease', 'Gene')

    elif 'Phenotype' in columns and 'Gene' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Phenotype', 'Gene')

    elif 'Drug' in columns and 'Gene' in columns:    
         
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Gene', 'Drug')

    elif 'Disease' in columns and 'Drug' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Disease', 'Drug')

    elif 'Disease' in columns and 'Phenotype' in columns:
         
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Disease', 'Phenotype')

    elif 'Drug' in columns and 'Phenotype' in columns:
        
        train_edge_index, train_labels, val_edge_index, val_labels, test_edge_index, test_labels, node_embeddings, node_enb_length = prepare_data(train_df, test_df, gene_embeddings, disease_embeddings, drug_embeddings, phenotype_embeddings, 'Phenotype', 'Drug')




    embedding_dim = 768
    output_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_embedding_nn = NodeEmbeddingNN(embedding_dim, output_dim).to(device)
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float).to(device)
    x = node_embedding_nn(node_embeddings)

    num_layers = 4
    model = GCN(in_channels=output_dim, hidden_channels=32, out_channels=64, num_layers=num_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = Data(x=x, edge_index=train_edge_index.to(device))
    test_data = Data(x=x, edge_index=test_edge_index.to(device))
    val_data = Data(x=x, edge_index=val_edge_index.to(device))

    for epoch in range(800):
        loss = train(train_data, train_edge_index, model, optimizer, criterion, node_enb_length, device)
        val_loss, val_acc, val_f1, val_roc_auc, val_aupr = validate(val_data, val_edge_index, model, criterion, node_enb_length, device)
        print(category_name)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val ROC AUC: {val_roc_auc:.4f}, Val AUPR: {val_aupr:.4f}')


    # test_loss, test_acc, test_f1, test_roc_auc, test_aupr = evaluate(test_data, test_edge_index, model, criterion)
    # print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}')
    # with open('heterograph_results.txt', 'a') as f:
    #     f.write(f'{category_name}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}\n')

    #test
    test_loss, test_acc, test_f1, test_roc_auc, test_aupr = evaluate_test_data(test_edge_index, model, criterion, x, node_enb_length, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}')
    with open('heterograph_results2.txt', 'a') as f:
        f.write(f'{category_name}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test AUPR: {test_aupr:.4f}\n')

