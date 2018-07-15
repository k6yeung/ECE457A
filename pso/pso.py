# Import modules
import numpy as np
import pyswarms as ps
import pandas as pd

df = pd.read_csv('dataset.csv')

# Decode dataset row from particle pos
def decode(position):
    out = 0
    for bit in position:
        out = (out << 1) | bit
    return out

# Accuracy of neural net given params
def accuracy(args):   
    result = np.ones(len(args), float)
    i = 0
    for pos in args:    
        row = decode(pos)
        if row < 24:
            # Return (1.0 - accuracy) since PSO optimizes for minima
            result[i] = 1.0-df.iloc[row].accuracy
        else:
            # Return highest accuracy value if row does not exist in dataset
            result[i] = 1.0
        i = i+1
    return result


def main():
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

    optimizer = ps.discrete.binary.BinaryPSO(n_particles=2, dimensions=5, options=options)

    # Perform optimization 
    cost, pos = optimizer.optimize(accuracy, print_step=25, iters=100, verbose=4)
    optimal_row = df.iloc[decode(pos)]
    print("Optimal Particle Position: " + str(pos))
    print("Optimal Neural Net Parameters :: # of Features: %s , # of Layers: %s , Learning Rate: %s" % (optimal_row.features, optimal_row.layers, optimal_row.learning_rate))
    print("Optimal Accuracy: " + str(1.0-cost))
    
    

if __name__ == "__main__":
    main()