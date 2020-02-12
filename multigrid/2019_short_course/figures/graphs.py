import networkx as nx
import matplotlib.pyplot as plt

savePlots = 1
showPlots = 1


# G = nx.random_geometric_graph(30, 0.3)
# G = nx.connected_caveman_graph(4,9)
G = nx.relaxed_caveman_graph(4,9,0.2,seed=42)
# G = nx.karate_club_graph()


nx.draw(G,node_size=100)

if savePlots:
   filename = 'graph.png'
   plt.savefig(filename, bbox_inches="tight")

######### Show the plots #########

if showPlots:
   plt.show()


