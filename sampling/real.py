import pandas as pd
from networkx import DiGraph, read_edgelist, set_edge_attributes

from functions import required_length, sample_datarow, build_epinions_digraph_from_local

def main():
    # graph = build_epinions_digraph_from_local()
    graph = read_edgelist(
        # "rdata/graph.csv",
        # "rdata/graph2.csv",
        # "rdata/graph3csv",
        "rdata/nostr_graph.csv",
        nodetype=int,
        delimiter=",",
        create_using=DiGraph
    )

    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    set_edge_attributes(graph, 1.0, "capacity")

    num_rows = 50
    num_file = 5
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
        data.to_csv(f"rdata/computation{str(i).zfill(2)}.csv")
        print("end iteration: ", i)


if __name__ == "__main__":
    main()
