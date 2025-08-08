import os
import time

from glob import glob
import networkx as nx
import pandas as pd
from numpy.random import RandomState

# from computation.graph import extended_barabasi_albert_digraph
from functions import required_length, sample_datarow
from test_function import test_function


def main():
    random_state = RandomState(42)

    n0 = 10  # initial nodes
    n = 100  # number of total nodes
    m = 2  # number of edges to attach from a new node
    p = 0.5  # probability of adding m new edges between existing nodes
    q = 0.3  # probability of rewiring edges

    # n0, n, m, p, q = 10, 200, 3, 0.3, 0.4
    # n0, n, m, p, q = 10, 1000, 3, 0.3, 0.4

    graph_fname = f"data/graph_{n0}_{n}_{m}_{p}_{q}.adjlist"
    print("Graph filename:", graph_fname)


    graph = nx.read_adjlist(graph_fname, nodetype=int, create_using=nx.DiGraph)
    nx.set_edge_attributes(graph, 1.0, "capacity")

    # if not os.path.exists(graph_fname):
    #     start = time.time()
    #     # Generate the graph with the defined parameters
    #     graph = extended_barabasi_albert_digraph(n0, n, m, p, q, random_state)
    #     nx.write_adjlist(graph, graph_fname)
    #     print("Graph generation time:", time.time() - start)
    #     nx.set_edge_attributes(graph, 1.0, "capacity")
    # else:
    #     graph = nx.read_adjlist(graph_fname, nodetype=int, create_using=nx.DiGraph)
    #     nx.set_edge_attributes(graph, 1.0, "capacity")

    # print(graph.number_of_nodes())
    # print(graph.number_of_edges())

    num_rows = 1000
    num_file = 100
    r_l = required_length(5, 0.85)
    for i in range(num_file):
        print("start iteration: ", i)
        data = [
            (print(_), sample_datarow(graph, n_bots=100, req_length=r_l))[1]
            for _ in range(num_rows)
        ]
        data = pd.DataFrame(data)
        # data.head(5)
        # 30 min per iterations
        data.to_csv(f"data/computation{str(i).zfill(2)}.csv")
        print("end iteration: ", i)
    test_function("data/computation*.csv")


if __name__ == "__main__":
    main()
