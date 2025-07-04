import math
import os
import time
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from computation import (compute_wot_score,
                         compute_wot_score_rev, personalized_random_walk)


def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


def extended_barabasi_albert_digraph(n0:int, n:int, m:int, p:int, q:int, 
                                     seed: Optional[np.random.RandomState] = None):
    """
    Generate an extended Barabási-Albert graph with additional functionality.
    
    Parameters:
        n0 (int): Initial number of nodes.
        n (int): Total number of nodes in the graph.
        m (int): Number of edges to add for each operation.
        p (float): Probability of adding m new edges between existing nodes.
        q (float): Probability of rewiring m edges.
        seed (np.random.RandomState, optional): Random number generator.

    Returns:
        nx.DiGraph: The generated graph.
    """
    if seed is None:
        seed = np.random.RandomState()

    if not (n0 >= 1 and m >= 1 and m < n and n0 < n):
        raise ValueError("Invalid values for n0, n, or m.")
    if not (0 <= p + q < 1):
        raise ValueError("Probabilities p and q must satisfy p + q < 1.")
    
    # Start with n0 initial nodes
    G:nx.DiGraph = nx.DiGraph()
    attachment_preference: List[int] = list(range(n0))
    G.add_nodes_from(attachment_preference)

    # Complete the graph 
    while G.number_of_nodes() < n:
        action_prob = seed.random()

        # Adding m new edges between existing nodes
        if action_prob < p: 
            for _ in range(m):
                while True:
                    src_node = seed.randint(0, G.number_of_nodes())
                    dst_node = seed.choice(attachment_preference)
                    if src_node != dst_node and not G.has_edge(src_node, dst_node):
                        # add new edge and increase the corresponding pref. attachment
                        G.add_edge(src_node, dst_node)
                        attachment_preference.append(dst_node)
                        break

        # Rewiring m edges
        elif p <= action_prob < (p + q):
            for _ in range(m):
                sd = list(G.edges())
                while True:
                    src_node, dst_node = sd[ seed.randint(0, len(sd)) ]
                    new_dst_node = seed.choice(attachment_preference)
                    if new_dst_node not in [src_node, dst_node] and not G.has_edge(src_node, new_dst_node):
                        # Rewire and adjust the preferential attachment list
                        G.remove_edge(src_node, dst_node)
                        G.add_edge(src_node, new_dst_node)
                        attachment_preference.remove(dst_node)
                        attachment_preference.append(new_dst_node)
                        break

        # Adding new node with m edges
        else:
            # Select the edges' nodes by preferential attachment
            new_node = G.number_of_nodes()
            targets = _random_subset(attachment_preference, m, seed)
            # G.add_edges_from(zip([new_node] * m, targets))
            G.add_edges_from((new_node, target) for target in targets)
            # 1 new attachment to the target, (m+1) attachments to the new_node
            attachment_preference.extend(list(targets) + [new_node] * (m + 1))
    nx.set_edge_attributes(G, 1.0, "capacity")
    
    return G


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
    FreeNet_pre_ranks_freenet = rankdata(FreeNet_pre_scores_freenet, method="min")

    FreeNet_post_scores_freenet = [
        compute_wot_score(graph_post, victim, trustee) for trustee in range(N)
    ]
    FreeNet_post_ranks_freenet = rankdata(FreeNet_post_scores_freenet, method="min")

    # Compute Reversed freenet WoT scores
    Reversed_pre_scores_freenet = [
        compute_wot_score_rev(graph_pre, victim, trustee) for trustee in range(N)
    ]
    Reversed_pre_ranks_freenet = rankdata(Reversed_pre_scores_freenet, method="min")

    Reversed_post_scores_freenet = [
        compute_wot_score_rev(graph_post, victim, trustee) for trustee in range(N)
    ]
    Reversed_post_ranks_freenet = rankdata(Reversed_post_scores_freenet, method="min")

    # # Compute Independent freenet WoT scores
    # Independent_pre_scores_freenet = [
    #     compute_wot_score_indep(graph_pre, victim, trustee) for trustee in range(N)
    # ]
    # Independent_pre_ranks_freenet = rankdata(Independent_pre_scores_freenet, method="min")

    # Independent_post_scores_freenet = [
    #     compute_wot_score_indep(graph_post, victim, trustee) for trustee in range(N)
    # ]
    # Independent_post_ranks_freenet = rankdata(Independent_post_scores_freenet, method="min")

    # Compute Random walk WoT scores
    Random_pre_scores_freenet = personalized_random_walk(
        graph_pre, np.random.RandomState(42), victim, req_length
    )
    Random_pre_ranks_freenet = rankdata(Random_pre_scores_freenet[:N], method="min")

    Random_post_scores_freenet = personalized_random_walk(
        graph_post, np.random.RandomState(42), victim, req_length
    )
    Random_post_ranks_freenet = rankdata(Random_post_scores_freenet[:N], method="min")

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
        # "Independent_pre_score": Independent_pre_scores_freenet[villain],
        # "Independent_post_score": Independent_post_scores_freenet[villain],
        # "Independent_pre_rank": Independent_pre_ranks_freenet[villain],
        # "Independent_post_rank": Independent_post_ranks_freenet[villain],
        # "Independent_delta_ranks": Independent_post_ranks_freenet[villain]
        # - Independent_pre_ranks_freenet[villain],
        "Random_pre_score": Random_pre_scores_freenet[villain],
        "Random_post_score": Random_post_scores_freenet[villain],
        "Random_pre_rank": Random_pre_ranks_freenet[villain],
        "Random_post_rank": Random_post_ranks_freenet[villain],
        "Random_delta_ranks": Random_post_ranks_freenet[villain]
        - Random_pre_ranks_freenet[villain],
        "dist_victim_relayer": get_distance_in_graph(graph_pre, victim, relayer),
        "dist_victim_vilain_pre": get_distance_in_graph(graph_pre, victim, villain),
        "dist_victim_vilain_post": get_distance_in_graph(graph_post, victim, villain),
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
    n = 1000  # number of total nodes
    m = 2  # number of edges to attach from a new node
    p = 0.5  # probability of adding m new edges between existing nodes
    q = 0.3  # probability of rewiring edges

    # n0, n, m, p, q = 10, 200, 3, 0.3, 0.4
    # n0, n, m, p, q = 10, 1000, 3, 0.3, 0.4

    graph_fname = f"bdata/graph_{n0}_{n}_{m}_{p}_{q}.adjlist"
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
        data.to_csv(f"bdata/computation{str(i).zfill(2)}.csv")
        print("end iteration: ", i)


if __name__ == "__main__":
    main()
