from networkx.algorithms import bipartite
import networkx as nx
import random
import timeit
import time
import os

import os

directory_path = 'genGraphs'

# Check if the directory already exists
if not os.path.exists(directory_path):
    # If it doesn't exist, create it
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")

fileName = "genGraphs/bipartiteGraph"
numOfGraphs = 5
n = 150
m = n
p = 0.6
#seed = 1
directed = False

for i in range(numOfGraphs):
    file = open(fileName+str(i)+".txt","a")
    seed = i
    # Generate a random graph
    # Record the start time
    start_time = time.time()
    g = nx.bipartite.random_graph(n, m, p, seed, directed)
    # Record the stop time
    stop_time = time.time()
    # Calculate the elapsed time
    elapsed_time = stop_time - start_time
    print(f"Generated Bipartite Graph in {elapsed_time} seconds")


    file.write(str(len(g.nodes()))+" "+str(len(g.edges()))+"\n")
    for j in g.edges():
        file.write(str(j[0]+1)+" "+str(j[1]+1)+"\n")
    file.write("\n")