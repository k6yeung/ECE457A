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
        # Retrieve dataset row encoding from particle pos
        out = 0
        for bit in pos:
            out = (out << 1) | bit
        # Return 1.0 - accuracy since PSO optimizes for minima
        if out < 24 :
            result[i] = 1.0-df.iloc[out].accuracy
        else :
            result[i] = 1.0
        i = i+1
    return result


def main():
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

    optimizer = ps.discrete.binary.BinaryPSO(n_particles=2, dimensions=5, options=options)
    
    # Perform optimization 
    cost, pos = optimizer.optimize(accuracy, print_step=100, iters=1000, verbose=4)

    print("Optimal Particle Position: " + str(pos))
    print("Optimal Accuracy: " + str(1.0-cost))
    
    

if __name__ == "__main__":
    main()