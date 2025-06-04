import math
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from computation import (compute_wot_score, compute_wot_score_indep,
                         compute_wot_score_rev, personalized_random_walk)
from graph import extended_barabasi_albert_digraph


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


def get_distance_in_graph(trust_graph, truster, trustee):
    try:
        dist = nx.shortest_path_length(trust_graph, source=truster, target=trustee)
    except nx.NetworkXNoPath:
        dist = -1  # float('inf')  # No path found
    return dist


def sample_datarow(graph, n_bots, req_length):
    graph_pre = graph.copy()

    N = graph_pre.number_of_nodes()
    bots_id_start = graph_pre.number_of_nodes()
    for i in range(n_bots):
        graph_pre.add_node(i + bots_id_start)
    bots_id_end = graph_pre.number_of_nodes()
    # print(bots_id_start, bots_id_end)

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
    graph_post.add_edge(relayer, villain, capacity=1.0)
    for bot in range(bots_id_start, bots_id_end):
        # relayer -> bots
        graph_post.add_edge(relayer, bot, capacity=1.0)
        # bots -> villain
        graph_post.add_edge(bot, villain, capacity=1.0)

    # Compute Freenet WoT scores
    FreeNet_pre_scores_freenet = [
        compute_wot_score(graph_pre, victim, trustee) for trustee in range(N)
    ]
    FreeNet_pre_ranks_freenet = rankdata(FreeNet_pre_scores_freenet)

    FreeNet_post_scores_freenet = [
        compute_wot_score(graph_post, victim, trustee) for trustee in range(N)
    ]
    FreeNet_post_ranks_freenet = rankdata(FreeNet_post_scores_freenet)

    # Compute Reversed freenet WoT scores
    Reversed_pre_scores_freenet = [
        compute_wot_score_rev(graph_pre, victim, trustee) for trustee in range(N)
    ]
    Reversed_pre_ranks_freenet = rankdata(Reversed_pre_scores_freenet)

    Reversed_post_scores_freenet = [
        compute_wot_score_rev(graph_post, victim, trustee) for trustee in range(N)
    ]
    Reversed_post_ranks_freenet = rankdata(Reversed_post_scores_freenet)

    # Compute Independent freenet WoT scores
    Independent_pre_scores_freenet = [
        compute_wot_score_indep(graph_pre, victim, trustee) for trustee in range(N)
    ]
    Independent_pre_ranks_freenet = rankdata(Independent_pre_scores_freenet)

    Independent_post_scores_freenet = [
        compute_wot_score_indep(graph_post, victim, trustee) for trustee in range(N)
    ]
    Independent_post_ranks_freenet = rankdata(Independent_post_scores_freenet)

    # Compute Random walk WoT scores
    Random_pre_scores_freenet = personalized_random_walk(
        graph_pre, np.random.RandomState(42), victim, req_length
    )
    Random_pre_ranks_freenet = rankdata(Random_pre_scores_freenet[:N])

    Random_post_scores_freenet = personalized_random_walk(
        graph_post, np.random.RandomState(42), victim, req_length
    )
    Random_post_ranks_freenet = rankdata(Random_post_scores_freenet[:N])

    datarow = {
        "FreeNet_pre_score": FreeNet_pre_scores_freenet[villain],
        "FreeNet_post_score": FreeNet_post_scores_freenet[villain],
        "FreeNet_pre_rank": FreeNet_pre_ranks_freenet[villain],
        "FreeNet_post_rank": FreeNet_post_ranks_freenet[villain],
        "FreeNet_delta_ranks": FreeNet_post_ranks_freenet[villain]
        - FreeNet_pre_ranks_freenet[villain],
        "Reversed_pre_score": Reversed_pre_scores_freenet[villain],
        "Reversed_post_score": Reversed_post_scores_freenet[villain],
        "Reversed_pre_rank": Reversed_pre_ranks_freenet[villain],
        "Reversed_post_rank": Reversed_post_ranks_freenet[villain],
        "Reversed_delta_ranks": Reversed_post_ranks_freenet[villain]
        - Reversed_pre_ranks_freenet[villain],
        "Independent_pre_score": Independent_pre_scores_freenet[villain],
        "Independent_post_score": Independent_post_scores_freenet[villain],
        "Independent_pre_rank": Independent_pre_ranks_freenet[villain],
        "Independent_post_rank": Independent_post_ranks_freenet[villain],
        "Independent_delta_ranks": Independent_post_ranks_freenet[villain]
        - Independent_pre_ranks_freenet[villain],
        "Random_pre_score": Random_pre_scores_freenet[villain],
        "Random_post_score": Random_post_scores_freenet[villain],
        "Random_pre_rank": Random_pre_ranks_freenet[villain],
        "Random_post_rank": Random_post_ranks_freenet[villain],
        "Random_delta_ranks": Random_post_ranks_freenet[villain]
        - Random_pre_ranks_freenet[villain],
        "dist_victim_relayer": get_distance_in_graph(graph_pre, victim, relayer),
        "out_deg_victim": graph_pre.out_degree(victim),
        "in_deg_villain": graph_pre.in_degree(villain),
        "victim": victim,
        "villain": villain,
        "relayer": relayer,
    }

    return datarow


def main():
    random_state = np.random.RandomState(42)

    n0 = 10  # initial nodes
    n = 100  # number of total nodes
    m = 2  # number of edges to attach from a new node
    p = 0.5  # probability of adding m new edges between existing nodes
    q = 0.3  # probability of rewiring edges

    # n0, n, m, p, q = 10, 200, 3, 0.3, 0.4
    # n0, n, m, p, q = 10, 1000, 3, 0.3, 0.4

    graph_fname = f"data/graph_{n0}_{n}_{m}_{p}_{q}.adjlist"
    print("Graph filename:", graph_fname)

    if not os.path.exists(graph_fname):
        start = time.time()
        # Generate the graph with the defined parameters
        graph = extended_barabasi_albert_digraph(n0, n, m, p, q, random_state)
        nx.write_adjlist(graph, graph_fname)
        print("Graph generation time:", time.time() - start)
        nx.set_edge_attributes(graph, 1.0, "capacity")
    else:
        graph = nx.read_adjlist(graph_fname, nodetype=int, create_using=nx.DiGraph)
        nx.set_edge_attributes(graph, 1.0, "capacity")

    num_rows = 1000
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
        data.to_csv(f"data/computation{i}.csv")
        print("end iteration: ", i)


if __name__ == "__main__":
    main()
