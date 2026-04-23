import dpctl
import sigmo
from sigmo.graph import chain_graph
import networkx as nx

dev = dpctl.SyclDevice("cuda:gpu")

#graphs = [chain_graph(20)]  
#graph2 = [chain_graph(30)]

G = nx.Graph()

G.add_node(1)
G.add_node(2)

G.add_edge(1, 2)

print(G)

#print(dev)

#SIGMOGRAPH = sigmo.generate_csr_signatures(dev, graphs, "data")
#print(SIGMOGRAPH)

#SIGMOGRAPH = sigmo.refine_csr_signatures(dev, graph2, "query", 1)
#print(SIGMOGRAPH)


#print(sigmo.generate_csr_signatures(dev, graphs, "data"))
#print(sigmo.generate_csr_signatures(dev, graph2, "query"))
#print(sigmo.refine_csr_signatures(dev, graphs, "data", 3))
#print(sigmo.refine_csr_signatures(dev, graph2, "query", 1))