import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import cluster, pairwise_distances, normalized_mutual_info_score, silhouette_score

from tqdm import tqdm

from FlagRep0 import truncate_svd, chordal_distance
from FlagRep import FlagRep



def set_seed(seed):
    np.random.seed(seed)                   
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def make_Bs(fl_type):
    Bs = [np.arange(fl_type[0])]
    for i in range(1,len(fl_type)):
        Bs.append(np.arange(fl_type[i-1],fl_type[i]))
    return Bs


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        # Input layer to hidden layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Hidden layer 2 to hidden layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Hidden layer 3 to output layer (10 classes)
        self.fc4 = nn.Linear(hidden_size, output_size)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on the output (for logits)
        return x

if __name__ == '__main__':

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Set up hyperparameters
    input_size = 28 * 28  # Flattened image size
    hidden_size = 128     # Same size for all hidden layers
    output_size = 10      # 10 output classes (digits 0-9)
    num_epochs = 10
    learning_rate = 0.001

    results_clustering = pd.DataFrame(columns = ['Method', 'n Clusters', 'Cluster Purity', 'Normalized Mutual Information', 'Silhouette Score'])

    for seed_num in range(20):
        set_seed(seed_num)
        print()
        print()
        print(f'trial {seed_num}')
        

        # Initialize the model, loss function, and optimizer
        model = NeuralNetwork(input_size, hidden_size, output_size)
        
        model.load_state_dict(torch.load(f'../models/mnist_model{seed_num}.pth'))
        
        # Pass the sample through the network step-by-step without computing gradients
        test_mats = []
        for img,_ in test_dataset:
            sample_image = img.view(-1, 28*28)
            with torch.no_grad():
                # Step-by-step pass through each layer
                h10 = model.fc1(sample_image)
                h11 = model.relu(h10)  # Pass through first hidden layer
                h20 = model.fc2(h11)
                h21 = model.relu(h20)  # Pass through second hidden layer
                h30 = model.fc3(h21)
                h31 = model.relu(h30)  # Pass through third hidden layer

                # hidden_rep = [h.cpu().detach().numpy().T for h in [h10,h11,h20,h21,h30,h31]]
                hidden_rep = [h.cpu().detach().numpy().T for h in [h31,h30,h21,h20]]
            test_mats.append(np.hstack(hidden_rep))


        As = [np.arange(2),np.arange(4)]
        flag_type = [2,4]


        flags = {}
        flags['FlagRep'] = []
        flags['QR'] = []
        flags['SVD'] = []

        flag_types = {}
        flag_types['FlagRep'] = []
        flag_types['QR'] = []
        flag_types['SVD'] = []

        for m in tqdm(test_mats):
            my_flag_rep = FlagRep(As, flag_type=[1,2])
            frep = my_flag_rep.fit_transform(m)
            flags['FlagRep'].append(frep)
            flag_types['FlagRep'].append([1,2])

            q, _ = np.linalg.qr(m)
            flags['QR'].append(q[:,:2])
            flag_types['QR'].append([1,2])

            u = truncate_svd(m)
            flags['SVD'].append(u[:,:2])
            flag_types['SVD'].append([1,2])
        

        n_pts = 100
        dist_mats = {}
        for method_name in ['FlagRep', 'QR', 'SVD']:
            #make distance matrices
            dist_mats[method_name] = np.zeros((n_pts,n_pts))
            for i in tqdm(range(n_pts)):
                for j in range(i+1,n_pts):
                    x = flags[method_name][i]
                    y = flags[method_name][j]
                    if method_name == 'Euclidean':
                        dist = np.linalg.norm(x-y)
                    else:
                        fl_type_x = flag_types[method_name][i]
                        fl_type_y = flag_types[method_name][j]
                        Bs_x = make_Bs(fl_type_x)
                        Bs_y = make_Bs(fl_type_y)
                        dist = chordal_distance(x, y, Bs_x, Bs_y)
                    dist_mats[method_name][i,j] = dist
                    dist_mats[method_name][j,i] = dist
                        

        mod_labels = [l for _, l in test_dataset][:n_pts]



        test_mats_baseline = []
        for img,_ in test_dataset:
            sample_image = img.view(-1, 28*28)
            with torch.no_grad():
                h10 = model.fc1(sample_image)
                h11 = model.relu(h10)  
                h20 = model.fc2(h11)
                h21 = model.relu(h20)         
                h30 = model.fc3(h21)
                h31 = model.relu(h30)         

                hidden_rep = h31.cpu().detach().numpy()
            test_mats_baseline.append(hidden_rep)

        small_euc_data = np.vstack(test_mats_baseline)[:n_pts,:]

        dist_mats['Euclidean'] = pairwise_distances(small_euc_data, metric='euclidean')



        for i, method_name in enumerate(['FlagRep', 'QR', 'SVD', 'Euclidean']):


            # n_clusters = 10  # Define the number of clusters
            for n_clusters in range(5,16):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
                cluster_labels = clustering.fit_predict(dist_mats[method_name])
                purity = purity_score(mod_labels, cluster_labels)
                nMI = normalized_mutual_info_score(mod_labels, cluster_labels)
                sil_score = silhouette_score(dist_mats[method_name], cluster_labels, metric='precomputed')


                row = pd.DataFrame(columns = results_clustering.columns, data = [[method_name, n_clusters, purity, nMI, sil_score]])
                results_clustering = pd.concat([results_clustering, row])

        results_clustering.to_csv('../results/MNIST_res1.csv')

    plt.figure(figsize = (9,3))
    sns.lineplot(results_clustering, y = 'Cluster Purity', x='n Clusters', hue = 'Method')
    plt.tight_layout()
    plt.show()
    # plt.savefig('../results/mnist20.pdf', bbox_inches = 'tight')
