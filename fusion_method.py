import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


class GeneDiseaseDataset(Dataset):
    def __init__(self, dataframe, gene_embeddings, disease_embeddings, phenotype_embeddings=None, drug_embeddings=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing 'id', 'Genes', and 'Association'.
            gene_embeddings (dict): A dictionary with gene IDs as keys and embeddings as values.
            disease_embeddings (dict): A dictionary with disease IDs as keys and embeddings as values.
            phenotype_embeddings (dict, optional): A dictionary with phenotype IDs as keys and embeddings as values.
            drug_embeddings (dict, optional): A dictionary with drug IDs as keys and embeddings as values.
        """
        self.dataframe = dataframe
        self.gene_embeddings = gene_embeddings
        self.disease_embeddings = disease_embeddings
        self.phenotype_embeddings = phenotype_embeddings
        self.drug_embeddings = drug_embeddings
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        
        columns = self.dataframe.columns
        if 'Disease' in columns and 'Gene' in columns:
            id1 = self.dataframe.iloc[idx]['Disease']
            embedding1 = self.disease_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Gene']
            embedding2 = self.gene_embeddings[id2]
        elif 'Phenotype' in columns and 'Gene' in columns:
            id1 = self.dataframe.iloc[idx]['Phenotype']
            embedding1 = self.phenotype_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Gene']
            embedding2 = self.gene_embeddings[id2]
        elif 'Drug' in columns and 'Gene' in columns:
            id1 = self.dataframe.iloc[idx]['Drug']
            embedding1 = self.drug_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Gene']
            embedding2 = self.gene_embeddings[id2]
        elif 'Disease' in columns and 'Drug' in columns:
            id1 = self.dataframe.iloc[idx]['Disease']
            embedding1 = self.disease_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Drug']
            embedding2 = self.drug_embeddings[id2]
        elif 'Disease' in columns and 'Phenotype' in columns:
            id1 = self.dataframe.iloc[idx]['Disease']
            embedding1 = self.disease_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Phenotype']
            embedding2 = self.phenotype_embeddings[id2]
        elif 'Drug' in columns and 'Phenotype' in columns:
            id1 = self.dataframe.iloc[idx]['Drug']
            embedding1 = self.drug_embeddings[id1]
            id2 = self.dataframe.iloc[idx]['Phenotype']
            embedding2 = self.phenotype_embeddings[id2]
        else:
            raise ValueError("Unexpected column names in the dataframe.")
        
        label = self.dataframe.iloc[idx]['Association']
        
        # Concatenate embeddings
        combined_embedding = np.concatenate([embedding1, embedding2])
        
        # Convert to PyTorch tensors
        combined_embedding_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return combined_embedding_tensor, label_tensor
        
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal Loss for binary classification.
        Parameters:
            alpha (float): Weighting factor for the positive class (usually < 0.5).
            gamma (float): Modulating factor to adjust the rate at which easy examples are down-weighted.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.
        Parameters:
            inputs (tensor): Logits predicted by the model.
            targets (tensor): True labels.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

        

class GeneDiseaseNN(nn.Module):
    def __init__(self, input_dim):
        super(GeneDiseaseNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim) 
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(0.2) 
        
        self.fc2 = nn.Linear(input_dim, 768) 
        self.bn2 = nn.BatchNorm1d(768)
        # self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(768, 192)  
        self.bn3 = nn.BatchNorm1d(192)
        # self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(192, 96)  
        self.bn4 = nn.BatchNorm1d(96)
        
        self.fc5 = nn.Linear(96, 48) 
        self.bn5 = nn.BatchNorm1d(48)
        
        self.fc6 = nn.Linear(48, 24) 
        self.bn6 = nn.BatchNorm1d(24)
        
        self.fc7 = nn.Linear(24, 1)  

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        # x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        
        x = torch.sigmoid(self.fc7(x))  
        return x








def train_model(model, train_loader, device, optimizer, criterion, num_epochs, val_loader, category_name, patience=3):
    model.train()

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }


    best_val_f1 = 0
    epochs_no_improve = 0
    early_stop = False


    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc, train_f1 = 0, 0, 0
        val_loss, val_acc, val_f1 = 0, 0, 0
        total_labels, total_predictions = [], []

        total_loss = 0
        all_labels = []
        all_predictions = []
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy() > 0.5)
            # print(f'loss: {loss.item()}')
            # print('labels')
            # display(labels)
            # print('outputs')
            # display(outputs)
            # import time
            # time.sleep(500000)
        print(len(all_predictions))
        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        history['train_loss'].append(total_loss/len(train_loader))
        history['train_acc'].append(acc)
        history['train_f1'].append(f1)


        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {acc}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}")
        # save model
        torch.save(model.state_dict(), 'model/model.pth')
        with open('training_results_fusion.txt', 'a') as f:
            if epoch == 0:
                f.write(f"Training results for {category_name}\n")
            f.write(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {acc}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}\n")

        val_total_loss, val_acc, val_f1,_,_ = evaluate_model(model, val_loader, device, criterion, category_name, test = False)
        history['val_loss'].append(val_total_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)


        if val_f1 >= best_val_f1 and epoch >18:
            best_val_f1 = val_f1
            epochs_no_improve= 0
            # torch.save(model.state_dict(), f'model/model_sampling_{category_name}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience and epoch >18:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            early_stop = True
            break


    if not early_stop:
            print('Training completed without early stopping.')


    return history




def evaluate_model(model, test_loader, device, criterion, name, test = False):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    acc = accuracy_score(all_labels, all_scores > 0.5)
    f1 = f1_score(all_labels, all_scores > 0.5)
    roc_auc = roc_auc_score(all_labels, all_scores)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    aupr = auc(recall, precision)

    if test == True:

        # Plotting the ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic {name}')
        plt.legend(loc="lower right")

        # Plotting the Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {aupr:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve {name}')
        plt.legend(loc="lower left")
        plt.tight_layout()
        # plt.show()
        # plt.savefig(f'/data/macaulay/Gene_disease/datas/COSMIC/plots/roc_pr_{name}.png')


    

        print(f"Test Loss: {total_loss/len(test_loader)}, Accuracy: {acc}, F1-Score: {f1}, ROC-AUC: {roc_auc}, AUPR: {aupr}")
        with open('test_results_fusion.txt', 'a') as f:
            
            f.write(f"Test Loss {name}: {total_loss/len(test_loader)}, Accuracy: {acc}, F1-Score: {f1}, ROC-AUC: {roc_auc}, AUPR: {aupr}\n")

    else:
        print(f"Validation Loss: {total_loss/len(test_loader)}, Accuracy: {acc}, F1-Score: {f1}, ROC-AUC: {roc_auc}, AUPR: {aupr}")

        return total_loss/len(test_loader), acc, f1, roc_auc, aupr

def plot_metrics(history, name):
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title(f'Training and Validation Accuracy {name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'bo-', label='Training F1')
    plt.plot(epochs, history['val_f1'], 'ro-', label='Validation F1')
    plt.title(f'Training and Validation F1 Score {name}')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f'/data/macaulay/Gene_disease/datas/COSMIC/plots/metrics_{name}.png')



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

#shufle training df and take 10% of it as validation, in diff variable
for train_df, test_df, category_name in zip(train_df_list, test_df_list, category_names):
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_dataset = GeneDiseaseDataset(train_df, gene_embeddings, disease_embeddings, phenotype_embeddings, drug_embeddings)
    val_dataset = GeneDiseaseDataset(val_df, gene_embeddings, disease_embeddings, phenotype_embeddings, drug_embeddings)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_dataset = GeneDiseaseDataset(test_df, gene_embeddings, disease_embeddings, phenotype_embeddings, drug_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = 768 + 768
    # Initialize the neural network model
    model = GeneDiseaseNN(input_dim)
    model.train()  # Set the model to training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    # criterion = nn.BCELoss()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 80

    history = train_model(model, train_loader, device, optimizer, criterion, num_epochs, val_loader, category_name)
    plot_metrics(history, category_name)
    evaluate_model(model, test_loader, device, criterion, category_name, test=True)


