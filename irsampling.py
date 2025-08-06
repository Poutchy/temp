import math

from igraph import Graph
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from computation_igraph import (compute_wot_score, compute_wot_score_rev, personalized_random_walk)

def read_adj_list_adjlist(filename: str) -> tuple[int, list[tuple[int, int]]]:
    edges = []
    with open(filename, 'r') as f:
        max_node = 0
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            src = int(parts[0])
            neighbors = [int(nb) for nb in parts[1:]]
            if len(neighbors) != 0:
                for dst in [int(nb) for nb in neighbors]:
                    edges.append((src, dst))
                    mn = max(neighbors)
            else:
                mn = 0
            max_node = max(max_node, src, mn)
    return max_node + 1, edges

def create_directed_graph_adjlist(filename: str) -> Graph:
    nb_vertex, edge_list = read_adj_list_adjlist(filename)
    g = Graph(directed=True)
    g.add_vertices(nb_vertex)
    g.add_edges(edge_list)

    return g

def required_length(topK: int, alpha: float) -> int:
    """
    The function required_length() returns the length that the personalized walk
    has to reach for the Personalized Pagerank to achieve the specified precision.
    """
    a = float(alpha)
    c = 100.0
    N = 10000000.0  # Upper bound for the size of the graph.

    res = c / (1 - a) * math.pow(float(topK), a) * math.pow(N, 1 - a)
    return round(res)


def get_distance_in_graph(trust_graph: Graph, truster, trustee):
    return trust_graph.distances(source=truster, target=trustee)


def sample_datarow(graph, n_bots, req_length):
    graph_pre = graph.copy()

    N = graph_pre.vcount()
    bots_id_start = graph_pre.vcount()
    graph_pre.add_vertices(n_bots)
    bots_id_end = graph_pre.vcount()

    # select the actors
    user_ids = list(range(N))
    victim = np.random.choice(user_ids)
    user_ids.remove(victim)
    relayer = np.random.choice(user_ids)
    user_ids.remove(relayer)
    villain = np.random.choice(user_ids)
    user_ids.remove(villain)

    # print(victim, relayer, villain)

    graph_post = graph_pre.copy()
    # relayer -> villain
    graph_post.add_edge(relayer, villain)
    for bot in range(bots_id_start, bots_id_end):
        # relayer -> bots
        graph_post.add_edge(relayer, bot)
        # bots -> villain
        graph_post.add_edge(bot, villain)

    # Compute Freenet WoT scores
    FreeNet_pre_scores = [
        compute_wot_score(graph_pre, victim, trustee) for trustee in range(N)
    ]
    FreeNet_pre_ranks = rankdata(FreeNet_pre_scores, method="min")

    FreeNet_post_scores = [
        compute_wot_score(graph_post, victim, trustee) for trustee in range(N)
    ]
    FreeNet_post_ranks = rankdata(FreeNet_post_scores, method="min")

    print("freenet")
    print("  score")
    print("    pre", FreeNet_pre_scores)
    print("    pos", FreeNet_post_scores)
    print("  rank")
    print("    pre", FreeNet_pre_ranks)
    print("    pos", FreeNet_post_ranks)

    # Compute Reversed freenet WoT scores
    Reversed_pre_scores = [
        compute_wot_score_rev(graph_pre, victim, trustee) for trustee in range(N)
    ]
    Reversed_pre_ranks = rankdata(Reversed_pre_scores, method="min")

    Reversed_post_scores = [
        compute_wot_score_rev(graph_post, victim, trustee) for trustee in range(N)
    ]
    Reversed_post_ranks = rankdata(Reversed_post_scores, method="min")

    print("reversed")
    print("  score")
    print("    pre", Reversed_pre_scores)
    print("    pos", Reversed_post_scores)
    print("  rank")
    print("    pre", Reversed_pre_ranks)
    print("    pos", Reversed_post_ranks)

    # Compute Random walk WoT scores
    Random_pre_scores = personalized_random_walk(
        graph_pre, np.random.RandomState(42), victim, req_length
    )
    Random_pre_ranks = rankdata(Random_pre_scores[:N], method="min")

    Random_post_scores = personalized_random_walk(
        graph_post, np.random.RandomState(42), victim, req_length
    )
    Random_post_ranks = rankdata(Random_post_scores[:N], method="min")

    print("random")
    print("  score")
    print("    pre", Random_pre_scores)
    print("    pos", Random_post_scores)
    print("  rank")
    print("    pre", Random_pre_ranks)
    print("    pos", Random_post_ranks)

    datarow = {
        "FreeNet_pre_score": FreeNet_pre_scores[villain],
        "FreeNet_post_score": FreeNet_post_scores[villain],
        "FreeNet_pre_rank": FreeNet_pre_ranks[villain],
        "FreeNet_post_rank": FreeNet_post_ranks[villain],
        "FreeNet_delta_ranks": FreeNet_post_ranks[villain]
        - FreeNet_pre_ranks[villain],
        "Reversed_pre_score": Reversed_pre_scores[villain],
        "Reversed_post_score": Reversed_post_scores[villain],
        "Reversed_pre_rank": Reversed_pre_ranks[villain],
        "Reversed_post_rank": Reversed_post_ranks[villain],
        "Reversed_delta_ranks": Reversed_post_ranks[villain]
        - Reversed_pre_ranks[villain],
        "Random_pre_score": Random_pre_scores[villain],
        "Random_post_score": Random_post_scores[villain],
        "Random_pre_rank": Random_pre_ranks[villain],
        "Random_post_rank": Random_post_ranks[villain],
        "Random_delta_ranks": Random_post_ranks[villain]
        - Random_pre_ranks[villain],
        "dist_victim_relayer": get_distance_in_graph(graph_pre, victim, relayer),
        "dist_victim_vilain_pre": get_distance_in_graph(graph_pre, victim, villain),
        "dist_victim_vilain_post": get_distance_in_graph(graph_post, victim, villain),
        "out_deg_victim": graph_pre.outdegree()[victim],
        "in_deg_villain": graph_pre.indegree()[villain],
        "victim": victim,
        "villain": villain,
        "relayer": relayer,
    }

    return datarow


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
