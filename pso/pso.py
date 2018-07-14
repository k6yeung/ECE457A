# Import modules
import numpy as np
import pyswarms as ps
import pandas as pd

df = pd.read_csv('dataset.csv')

# Accuracy of neural net given params
def accuracy(args):   
    result = np.ones(len(args), float)
    i = 0
    for pos in args:    
        # Retrieve dataset row encoding of position
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        if pos <= 2 :
            result[i] = 100.0-df.iloc[out]
        else :
            result[i] = 100.0
        i = i+1
    # print(result)    
    return result


def main():
    # print(df.iloc[2])
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    optimizer = ps.discrete.binary.BinaryPSO(n_particles=2, dimensions=5, options=options)
    # Perform optimization 
    cost, pos = optimizer.optimize(accuracy, print_step=100, iters=100, verbose=4)

    

if __name__ == "__main__":
    main()