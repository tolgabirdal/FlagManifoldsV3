import torch

import pandas as pd

import sys
sys.path.append('../scripts')

import fl_algorithms as fl
import center_algorithms as ca


import numpy as np


from PSA_utils import *


if __name__ == '__main__':

    n_pcs = 5

    data = torch.load('../data/cats_dogs/myCATS.pt').numpy().T
    p, n = data.shape


    models = [[1,1,1,1,1,4096-n_pcs],
            [1,1,1,2,4096-n_pcs],[1,1,2,1,4096-n_pcs],[1,2,1,1,4096-n_pcs],[2,1,1,1,4096-n_pcs],
            [3,1,1,4096-n_pcs],[2,2,1,4096-n_pcs],[1,3,1,4096-n_pcs],[2,1,2,4096-n_pcs],[1,2,2,4096-n_pcs],[1,1,3,4096-n_pcs],
            [4,1,4096-n_pcs],[3,2,4096-n_pcs],[2,3,4096-n_pcs],[1,4,4096-n_pcs],
            [5,4096-n_pcs]]

    #regular PCA
    eigval, eigvec = evd(data)
    baseline_weights = eigvec[:,:n_pcs]
    true_fl_type, _ = model_selection_eval(data, eigval, models, criterion="bic")
    true_fl_type = list(np.cumsum(true_fl_type)[:-1])



    n_splits = list(range(2,7,1))
    n_trials = 20

    results = pd.DataFrame(columns = ['Number of Splits','Trial', 'Distance', 'Average']) 

    for n_split in n_splits:

        print(f'running split {n_split}')
        for trial in range(n_trials):
            np.random.seed(trial)

            weight_data = []
            fl_types = []
            left_over = np.arange(p)
            for split in range(n_split):
                left_over = np.array(list(left_over))
                # subset data
                idx0 = np.random.choice(len(left_over), (p-1)//n_split, replace = False)
                idx = left_over[idx0]
                subset_data = data[idx, :]
                # left over idx
                left_over = set(left_over).difference(set(idx))
            

                # compute pca   
                eigval, eigvec = evd(subset_data)
                weights = eigvec[:,:n_pcs]
                weight_data.append(weights)

                fl_type0, bic0 = model_selection_eval(subset_data, eigval, models, criterion="bic")

                fl_type = np.cumsum(fl_type0)[:-1]
                fl_types.append(list(fl_type))
            
            
            weight_data_stacked = np.stack(weight_data, axis = 2)

            # df_avg_weights, mean_f_type = fl.dynamic_flag_mean(weight_data_stacked, fl_types, verbosity=1)

            mean_f_type = most_common_fl_type(fl_types)

            #flag average of type mean
            f_avg_weights = fl.flag_mean(weight_data_stacked, mean_f_type, verbosity = 0)

            #flag average of type (1,2,3,...,n_pcs)
            f0_avg_weights = fl.flag_mean(weight_data_stacked, list(np.arange(1,mean_f_type[-1]+1)), verbosity = 0)

            #flag average of type (n_pcs)
            #equivalent to gr(n_pcs,n) flag mean
            g_avg_weights = ca.flag_mean(weight_data, mean_f_type[-1])
            #g_avg_weights = fl.flag_mean(weight_data_stacked, [3], verbosity = 0)

            #euclidean average
            e_avg_weights = np.mean(weight_data_stacked, axis = 2)
            e_avg_weights = np.linalg.qr(e_avg_weights)[0][:,:mean_f_type[-1]]


            #distances
            # df_dist = fl.chordal_dist(df_avg_weights, baseline_weights, true_fl_type)

            f_dist = fl.chordal_dist(f_avg_weights, baseline_weights, true_fl_type)

            f0_dist = fl.chordal_dist(f0_avg_weights, baseline_weights, true_fl_type)

            g_dist = fl.chordal_dist(g_avg_weights, baseline_weights, true_fl_type)

            e_dist = fl.chordal_dist(e_avg_weights, baseline_weights,true_fl_type)

            rand_pt = np.random.rand(n,mean_f_type[-1])-.5
            rand_pt = np.linalg.qr(rand_pt)[0][:,:mean_f_type[-1]]
            r_dist = fl.chordal_dist(rand_pt, baseline_weights, true_fl_type)


            trial_results = pd.DataFrame(columns = ['Number of Splits','Trial', 'Distance', 'Average'], 
                        data = [#[n_split, trial, df_dist, f'dFL-mean{mean_f_type}'],
                                [n_split, trial, f0_dist, f'FL-mean{[1,2,3,4,5]}'],
                                [n_split, trial, f_dist, f'FL-mean{mean_f_type}'],
                                [n_split, trial, g_dist, 'GR-mean'],
                                [n_split, trial, e_dist, 'Euclidean-mean'],
                                [n_split, trial, r_dist, 'Random']])
            results = pd.concat([results, trial_results])



            print(true_fl_type)
            print(mean_f_type)
            print()


            
            results.to_csv('../results/cats_dogs.csv')