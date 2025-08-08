import numpy as np
import pandas as pd
from igraph import Graph

from sampling.functions import required_length
from sampling.functions_igraph import sample_datarow


def main():
    random_state = np.random.RandomState(42)

    edges = []
    with open("rdata/graph4.csv", "r") as f:
        for line in f:
            source, target = map(int, line.strip().split(","))
            edges.append((source, target))

    # Create directed graph
    graph = Graph(edges=edges, directed=True)
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
        data.to_csv(f"idata/computation{str(i).zfill(2)}.csv")
        print("end iteration: ", i)


if __name__ == "__main__":
    main()
