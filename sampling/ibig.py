import numpy as np
import pandas as pd

from sampling.functions import required_length
from sampling.functions_igraph import (create_directed_graph_adjlist,
                                       sample_datarow)


def main():
    random_state = np.random.RandomState(42)

    n0 = 10  # initial nodes
    n = 1000  # number of total nodes
    m = 2  # number of edges to attach from a new node
    p = 0.5  # probability of adding m new edges between existing nodes
    q = 0.3  # probability of rewiring edges

    # n0, n, m, p, q = 10, 200, 3, 0.3, 0.4
    # n0, n, m, p, q = 10, 1000, 3, 0.3, 0.4

    graph_fname = f"bdata/graph_{n0}_{n}_{m}_{p}_{q}.adjlist"
    graph = create_directed_graph_adjlist(graph_fname)
    # print("start count of nodes", graph.vcount())

    print(graph.vcount())
    print(graph.ecount())

    num_rows = 100
    num_file = 10
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
        data.to_csv(f"idata/b/computation{str(i).zfill(2)}.csv")
        print("end iteration: ", i)


if __name__ == "__main__":
    main()
