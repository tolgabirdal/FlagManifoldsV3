import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import cluster, pairwise_distances
from sklearn.manifold import TSNE

from tqdm import tqdm

from FlagRep0 import truncate_svd, chordal_distance
from FlagRep import FlagRep

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

import pandas as pd


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

def get_latent_space_train(train_dataset, n_shots, model, seed0):
    np.random.seed(seed0)

    shot_ids = []
    train_lbls = np.array([lbl for _, lbl in train_dataset])
    train_dta = [img for img,_ in train_dataset] #bad coding
    for l in np.unique(train_lbls):
        idx = np.where(train_lbls == l)[0]
        idxc = np.random.choice(idx, n_shots,replace = False)
        shot_ids.append(idxc)

    shot_dict = {}
    euc_shot_dict = {}
    for i, s in enumerate(shot_ids):
        hidden_rep0s = []
        hidden_rep1s = []
        euc_shot_dict[i] = np.zeros((128,1))
        # euc_shot_dict[i] = np.zeros((256,1))
        for img in [train_dta[si] for si in s]:
            sample_image = img.view(-1, 28*28)
            with torch.no_grad():
                # Step-by-step pass through each layer
                h10 = model.fc1(sample_image)
                h11 = model.relu(h10)  # Pass through first hidden layer
                h20 = model.fc2(h11)
                h21 = model.relu(h20)         # Pass through second hidden layer
                h30 = model.fc3(h21)
                h31 = model.relu(h30)         # Pass through third hidden layer

                hidden_rep0 = h31.cpu().detach().numpy().T
                hidden_rep1 = h21.cpu().detach().numpy().T
                
                hidden_rep0s.append(hidden_rep0)
                hidden_rep1s.append(hidden_rep1)
                euc_shot_dict[i] += h31.cpu().detach().numpy().T
                # euc_shot_dict[i] += np.vstack([h31.cpu().detach().numpy().T,h21.cpu().detach().numpy().T])

        shot_dict[i] = np.hstack([np.hstack(hidden_rep0s),np.hstack(hidden_rep1s)])
        euc_shot_dict[i] = euc_shot_dict[i]/n_shots

    return shot_dict, euc_shot_dict


def get_train_flags(shot_dict, fl_type = [1,2]):
    shot_flags = {}
    shot_flags['FlagRep'] = []
    shot_flags['QR'] = []
    shot_flags['SVD'] = []
    for lbl, m in shot_dict.items():
        # my_flag_rep = FlagRep( [np.arange(2*n_shots), np.arange(4*n_shots)],
                            # flag_type=[1,2])
        my_flag_rep = FlagRep( [np.arange(n_shots), np.arange(2*n_shots)],
                            flag_type=fl_type)
        frep = my_flag_rep.fit_transform(m)
        shot_flags['FlagRep'].append(frep)

        q, _ = np.linalg.qr(m)
        shot_flags['QR'].append(q[:,:fl_type[-1]])

        u = truncate_svd(m)
        shot_flags['SVD'].append(u[:,:fl_type[-1]])

    return shot_flags

def get_latent_space_test(test_dataset):
    # Pass the sample through the network step-by-step without computing gradients
    test_mats = []
    euc_test = []
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
            # hidden_rep = [h.cpu().detach().numpy().T for h in [h31,h30,h21,h20]]
            hidden_rep = [h.cpu().detach().numpy().T for h in [h31,h21]]
        test_mats.append(np.hstack(hidden_rep))
        euc_test.append(h31.cpu().detach().numpy().T)
        # euc_test.append(np.vstack(hidden_rep))
    return test_mats, euc_test
        
def get_test_flags(test_mats, fl_type = [1,2]):
    flags = {}
    flags['FlagRep'] = []
    flags['QR'] = []
    flags['SVD'] = []
    for m in test_mats:
        my_flag_rep = FlagRep( [np.arange(i) for i in fl_type], 
                            flag_type=fl_type)
        frep = my_flag_rep.fit_transform(m)
        flags['FlagRep'].append(frep)

        q, _ = np.linalg.qr(m)
        flags['QR'].append(q[:,:fl_type[-1]])

        u = truncate_svd(m)
        flags['SVD'].append(u[:,:fl_type[-1]])

    return flags

def get_dist_mats(flags, Bs, euc_test, shot_flags, euc_shot_dict):
    dist_mats = {}
    for method_name in ['FlagRep', 'QR', 'SVD']:
        dist_mats[method_name] = np.zeros((10,len(flags[method_name])))
        for i, shot_flag in enumerate(shot_flags[method_name]):
            for j, flag in enumerate(flags[method_name]):
                dist_mats[method_name][i,j] = chordal_distance(shot_flag, flag, Bs, Bs)

    dist_mats['Euclidean'] = np.zeros((10,len(euc_test)))
    for i, shot in euc_shot_dict.items():
        for j, pt in enumerate(euc_test):
            dist_mats['Euclidean'][i,j] = np.linalg.norm(shot.T - pt.T)
    return dist_mats

def evaluate_trial(results, dist_mats, test_dataset, test_loader, num_shots):
    for method_name in ['FlagRep', 'QR', 'SVD', 'Euclidean', 'Baseline']:
        if method_name == 'Baseline':
            # Testing the model
            model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(-1, 28*28)  # Flatten images
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = correct / total
            row = pd.DataFrame(columns = results.columns,
                            data = [[method_name, acc, num_shots]])
            results = pd.concat([results, row])

        else:
            test_preds = np.argmin(dist_mats[method_name], axis = 0)
            test_labels = [lbl for _, lbl in test_dataset]
            acc = accuracy_score(test_labels, test_preds)
            row = pd.DataFrame(columns = results.columns,
                            data = [[method_name, acc, num_shots]])
            results = pd.concat([results, row])

    return results




if __name__ == '__main__':

    set_seed(42)

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Set up hyperparameters
    input_size = 28 * 28  # Flattened image size
    hidden_size = 128     # Same size for all hidden layers
    output_size = 10      # 10 output classes (digits 0-9)
    num_epochs = 10
    learning_rate = 0.001

    
    fl_type = [1,2]
    n_shots = 10

    results = pd.DataFrame(columns = ['Method', 'Accuracy', 'Number Shots'])


    for m_num in range(20):
        print(f'Starting model {m_num}')

        # Initialize the model, loss function, and optimizer
        model = NeuralNetwork(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(f'../models/mnist_model{m_num}.pth'))

        test_mats, euc_test = get_latent_space_test(test_dataset)

        flags = get_test_flags(test_mats)

        
        for n_shots in range(1,13,3):
            for seed0 in tqdm(range(5)):
            
                shot_dict, euc_shot_dict = get_latent_space_train(train_dataset, n_shots, model, seed0)

                shot_flags = get_train_flags(shot_dict)

                Bs = make_Bs(fl_type)

                dist_mats = get_dist_mats(flags, Bs, euc_test, shot_flags, euc_shot_dict)

                results = evaluate_trial(results, dist_mats, test_dataset, test_loader, n_shots)

            print(f'Num shots: {n_shots}')
            for method in np.unique(results['Method']):
                res = results[results['Method'] == method]['Accuracy'].mean()
                print(f'{method}: {res}')
            
            print()

    results.to_csv('../results/mnist_fewshot_idea.csv')

    

