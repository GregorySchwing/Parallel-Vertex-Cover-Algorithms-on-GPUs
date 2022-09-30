from networkx.algorithms import bipartite
import sys
from itertools import repeat, chain
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import scipy as sp

import scipy.io  # for mmread() and mmwrite()
import io  # Use BytesIO as a stand-in for a Python file object
"""
http://homepage.cs.uiowa.edu/~sriram/5350/fall19/notes/11.7/11.7.pdf

In fact, the output of the greedy algorithm can be quite poor, as showing in the following claim.
Claim: There exists a graph G = (V, E), |V | = n such that |S| = Ω(log(n)) · |S∗|, where S is the
output of greedy algorithms, and S∗ is the optimal vertex cover.

Proof: Construct a bipartite graph (Figure 2), which contains G = (L ∪ R, E). Let k denote |L|,
where R = R1 ∪ R2 ∪ · · · Rk. For each set Ri:

1. |Ri| = floor(k/i)
2. Each vertex in Ri has degree i and no two vertexes in Ri have a common neighbour.

The vertex in Rk would be the first node to to be selected to join S by the greedy MVC
algorithm since its degree is k. All other nodes in R have degree less than k and nodes in L also
have degree less than k (the largest degree in the L is k − 1). After deleting all the edges that
connected with this node Rk, all the degree of nodes in the L will also decrease 1. Then in the
next iterations, the nodes in Rk−1 will be selected one by one, and on and on until all the nodes
in R are selected to join S.
Then the final output |S| =Pki=2 |Ri| = bkic| ∼ kPki=11

i = Θ(k log(k)), but |L| = k. .
"""
def repeat_it(lst, numbers):
    return chain.from_iterable(repeat(i, j) for i, j in zip(lst, numbers))




# Instantiate the parser
parser = argparse.ArgumentParser(description='Create counter examples for greedy min vertex cover.')

# Required positional argument
parser.add_argument('-k', type=int,
                    help='k <The size of the min vertex cover>')

parser.add_argument('-o', type=str, nargs='?',
                    help='-o <The prefix for the output file>')
                    
parser.add_argument('-p', type=str, nargs='?',
                    help='-p <The prefix for the image file>')
  
args = parser.parse_args()

print("Argument values:")
print(args.k)
print(args.o)
print(args.p)

k = args.k
mtxPrefix = args.o
imagePrefix = args.p

B = nx.Graph()
numNodes = 1
# Create L set

# R_1

L = [x for x in range(1,k+1)]
print ("L == R_1: ",k)
print (L)
numNodes += len(L)
B.add_nodes_from(L, bipartite=0)
# R_1 == L
for x in range(2,k+1):
	print ("R_", x, ": ",k//x)
	R = [x for x in range(numNodes,numNodes+k//x)]
	print(R)
	numNodes += len(R)
	B.add_nodes_from(R, bipartite=1)
	zipped = zip(L, repeat_it(R, [(len(L)//len(R))] * len(R)))
	edgeList = list(zipped)
	print (edgeList)
	B.add_edges_from(edgeList)
	#B.add_edges_from([(1,26)])

if (mtxPrefix):
  fh = io.BytesIO()
  a = nx.to_scipy_sparse_array(B)
  sp.io.mmwrite(fh, a)

  # Write the stuff
  with open("{}.mtx".format(mtxPrefix), "wb") as f:
      f.write(fh.getbuffer())

if (imagePrefix):
  edge_widths = 0.1

  nx.draw_networkx(
      B,
      pos = nx.drawing.layout.bipartite_layout(B, L), 
      width = edge_widths*5) # Or whatever other display options you like

  # displaying the title
  plt.suptitle("Constructed bipartite graph counterexample for k-VC")
  plt.title("k =" + str(k) + "; n = " + str(numNodes) + "; Greedy solution = Ω(log(n)) · k")
  plt.savefig('{}.png'.format(imagePrefix), bbox_inches='tight')
      
  nx.drawing.nx_pydot.write_dot(B,'foo.dot')


