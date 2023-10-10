from networkx.algorithms import bipartite
import networkx as nx
import random
import timeit
import time
import os

import os
from pathlib import Path
directory_path = 'genGraphs'

# Check if the directory already exists
if not os.path.exists(directory_path):
    # If it doesn't exist, create it
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")

fileName = "bipartiteGraph"
numOfGraphs = 5


path2src = "/home/greg/Parallel-Vertex-Cover-Algorithms-on-GPUs/src"
path2GraphFolder = "/home/greg/Parallel-Vertex-Cover-Algorithms-on-GPUs/test/GraphGenerators/genGraphs"
# Define the base directory path
base_directory = Path(path2GraphFolder)
import os

import subprocess

# Replace 'binary_name' with the actual name or path of the binary you want to execute.
binary_name = './output'
# Replace 'arg1', 'arg2', etc., with the actual arguments you want to pass to the binary.

# Change the current working directory to the specified path
os.chdir(path2src)

for exponent in range(1, 6):  # Increase n by powers of 10 (1 to 4)
    n = 10 ** exponent
    m = n
    p = 0.6
    #seed = 1
    directed = False

    for i in range(numOfGraphs):
        fileNameOfThisGraph =  fileName+str(i)+"_"+str(n)+"_"+str(p)+".txt"
        print(str(base_directory))
        print(fileNameOfThisGraph)
        file = open(str(base_directory / fileNameOfThisGraph),"w")
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
        file.close()

        import subprocess

        # Replace 'arg1', 'arg2', etc., with the actual arguments you want to pass to the binary.
        args = ['-f', str(base_directory / fileNameOfThisGraph)]

        try:
            # Use subprocess.run to execute the binary with the specified arguments.
            result = subprocess.run([binary_name] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # You can access the binary's output and error messages like this:
            stdout = result.stdout
            stderr = result.stderr

            # Print the output and error messages if needed.
            print("Output:")
            print(stdout)

            print("Error:")
            print(stderr)

        except subprocess.CalledProcessError as e:
            # Handle errors if the binary returns a non-zero exit code.
            print(f"Error: {e}")

