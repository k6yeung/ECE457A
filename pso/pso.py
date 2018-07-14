# Import modules
import numpy as np
import pyswarms as ps
import pandas as pd

df = pd.read_csv('dataset.csv')

# Accuracy of neural net given params
def accuracy(args):
    print(args)    
    result = np.ones(len(args), float)
    i = 0
    for pos in args:
        res = df.loc[(df.features == pos[0]) & (df.layers == pos[1]) & (df.learning_rate == pos[2])]
        if res.empty :
            result[i] = 100.0
        else :
            result[i] = 100.0-res.accuracy.astype(float)
        i = i+1
    # print(result)    
    return result



def main():
    # Set-up hyperparameters
    options = {'c1': 0.002, 'c2': 0.001, 'w':1, 'k': 2, 'p': 2}

    # Set-up bounds of PSO determined by dataset values
    min_bound = np.ones(3)
    min_bound[0] = 2
    min_bound[1] = 3
    min_bound[2] = 0.001

    max_bound = np.ones(3)
    max_bound[0] = 7
    max_bound[1] = 3
    max_bound[2] = 0.003
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    # optimizer = ps.single.GlobalBestPSO(n_particles=2, dimensions=3, options=options, bounds=bounds)

    optimizer = ps.discrete.binary.BinaryPSO(n_particles=2, dimensions=3, options=options)
    # Perform optimization
    cost, pos = optimizer.optimize(accuracy, print_step=100, iters=100, verbose=4)

if __name__ == "__main__":
    main()