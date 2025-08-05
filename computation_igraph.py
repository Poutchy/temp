import json
import math

import numpy as np
from igraph import Graph

with open("config.json", "r", encoding="utf-8") as jsonfile:
    data = json.load(jsonfile)
    print("Read successful")


def compute_wot_rank(trust_graph: Graph, truster: int, trustee: int) -> int:
    """
    Compute the rank (shortest path distance for positive edges only) from truster to trustee.
    """
    # Compute shortest path length

    rank = trust_graph.distances(source=truster, target=trustee)
    # except ig:
    #     rank = -1  # float('inf')  # No path found
    return rank[0][0]


def compute_wot_capacity(rank: int) -> int:
    """
    Compute the capacity based on the rank using the defined exponential decay formula.
    """
    if rank < 0:
        return 0
    # if rank >= 100:
    #     return 0 # infinite

    if rank > 5:
        return 1

    return data["rank_to_capacity"][rank]


def compute_wot_score(
    trust_graph: Graph, truster: int, trustee: int, verbose: bool = False
) -> int:
    """
    Compute the score of trustee from the perspective of truster.
    """
    if truster == trustee:
        return 100
    if trustee in trust_graph.neighbors(truster, mode='out'):
        # Direct trust value exists
        # return 100 #trust_graph[truster][trustee]['weight']

        return 100
    # Indirect computation

    score = 0
    neighbor_of_trustee = trust_graph.neighbors(trustee, mode='in')
    # print("return of incident", neighbor_of_trustee)
    for neighbor in neighbor_of_trustee:
        trust_value = 100  # trust_graph[neighbor][trustee]['weight']
        rank = compute_wot_rank(trust_graph, truster, neighbor)
        capacity = compute_wot_capacity(rank)
        score += (trust_value * capacity) // 100

        if verbose:
            if rank >= 0:
                print(
                    f"  {truster}--...-->{neighbor}->{trustee}: rank={rank} capacity={capacity}"
                )
            else:
                print(
                    f"  {truster} NOPATH {neighbor}->{trustee}: rank={rank} capacity={capacity}"
                )
    # if score != 0:
    #     score = round(score / len(neighbor_of_trustee))

    return score


def compute_wot_score_rev(
    trust_graph: Graph, truster: int, trustee: int, verbose: bool = False
) -> int:
    """
    Compute the score of trustee from the perspective of truster.
    """
    # if truster == trustee:
    #     return 100
    # if trust_graph.has_edge(truster, trustee):
    #     # Direct trust value exists
    #     # return 100 #trust_graph[truster][trustee]['weight']

    #     return trust_graph.get_edge_data(truster, trustee).get("weight", 100)
    if truster == trustee:
        return 100
    if trustee in trust_graph.neighbors(truster, mode='out'):
        # Direct trust value exists
        # return 100 #trust_graph[truster][trustee]['weight']

        return 100
    # Indirect computation

    score = 0
    outgoing_of_truster = trust_graph.neighbors(truster, mode='out')
    for followed in outgoing_of_truster:
        trust_value = 100  # trust_graph[truster][followed]['weight']
        rank = compute_wot_rank(trust_graph, followed, trustee)
        capacity = compute_wot_capacity(rank)
        score += (trust_value * capacity) // 100

        if verbose:
            if rank >= 0:
                print(
                    f"  {truster}->{followed}--...-->{trustee}: rank={rank} capacity={capacity}"
                )
            else:
                print(
                    f"  {truster}->{followed} NOPATH {trustee}: rank={rank} capacity={capacity}"
                )
    # if score != 0:
    #     score = round(score / len(outgoing_of_truster))

    return score


# def compute_wot_score_indep(
#     trust_graph: ig.Graph, truster: int, trustee: int, verbose: bool = False
# ) -> int:
#     """
#     Compute the score of trustee from the perspective of truster with independent paths
#     """
#     if truster == trustee:
#         return 100
#     if trust_graph.has_edge(truster, trustee):
#         # Direct trust value exists
#         # return 100 #trust_graph[truster][trustee]['weight']

#         return trust_graph.get_edge_data(truster, trustee).get("weight", 100)
#     _, partition = nx.minimum_cut(trust_graph, truster, trustee)
#     partA, partB = partition
#     partA, partB = set(partA), set(partB)
#     truster_in_a = truster in partA
#     edge_cut = []
#     for s, t in trust_graph.edges():
#         if s in partA and t in partB and truster_in_a:
#             edge_cut.append((s, t))
#         if s in partB and t in partA and not truster_in_a:
#             edge_cut.append((t, s))
#     score = 0
#     for nodeA, nodeB in edge_cut:
#         d1 = nx.shortest_path_length(trust_graph, source=truster, target=nodeA)
#         d2 = nx.shortest_path_length(trust_graph, source=nodeB, target=trustee)

#         rank = d1 + d2
#         capacity = compute_wot_capacity(rank)
#         score += capacity
#     return score


rng = np.random.RandomState(42)  # Use the existing random_state variable


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


def personalized_random_walk(graph, rng, source_node, max_steps, alpha=0.85):
    """
    Perform a personalized random walk on the graph starting from a specific source node.

    Parameters:
        graph (nx.DiGraph): The input graph.
        source_node (int): The starting/teleporting node for the personalized random walk.
        max_steps (int): The maximum number of steps for the random walk.
        alpha (float): Dampening factor / teleport probability (default is 0.85).

    Returns:
        List[float]: The stationary distribution of the random walk.
    """
    # Initialize the probability distribution

    num_nodes = graph.vcount()
    prob = np.zeros(num_nodes)
    # prob[source_node] = 1  # Start from the source node

    node = source_node
    # Perform the random walk

    for _ in range(max_steps):
        if rng.random() > alpha:
            node = source_node  # reset
        # pick a random successor node from graph.node

        successors = list(graph.neighbors(node, mode='out'))
        if successors:
            node = rng.choice(successors)
            prob[node] += 1
        else:
            node = source_node  # Reset if no successors
    # normalize

    sum_prob = np.sum(prob)
    if sum_prob > 0:
        prob /= sum_prob
    # # Perform the random walk
    # for _ in range(max_steps):
    #     new_prob = np.zeros(num_nodes)
    #     for node in graph.nodes():
    #         for neighbor in graph.successors(node):
    #             new_prob[neighbor] += alpha * prob[node] / graph.out_degree(node)
    #         new_prob[node] += (1 - alpha) / num_nodes
    #     prob = new_prob

    return prob


def compare_score_ranks(g1: Graph, g2: Graph, truster: int, f1, f2, f3):
    # compute all scores and rev_scores, and compare the ranks using the Spearman correlation coefficient
    alls1 = []
    alls2 = []
    alls3 = []
    alla1 = []
    alla2 = []
    alla3 = []
    # for trustee in graph.nodes():
    for trustee in range(102):
        s1 = f1(g1, truster, trustee)
        s2 = f2(g1, truster, trustee)
        s3 = f3(g1, truster, trustee)
        a1 = f1(g2, truster, trustee)
        a2 = f2(g2, truster, trustee)
        a3 = f3(g2, truster, trustee)
        alls1.append(s1)
        alls2.append(s2)
        alls3.append(s3)
        alla1.append(a1)
        alla2.append(a2)
        alla3.append(a3)

    # Compute the Spearman correlation
    fin1 = np.corrcoef(alls1, alla1)[0, 1]
    fin2 = np.corrcoef(alls2, alla2)[0, 1]
    fin3 = np.corrcoef(alls3, alla3)[0, 1]
    return fin1, fin2, fin3


def compute_coef(g1, g2, f1, f2, f3):
    # Compute the Spearman correlation for all nodes
    coef1 = []
    coef2 = []
    coef3 = []
    for truster in [nodes for nodes in g1.nodes() if nodes < 102]:
        all_coefs = compare_score_ranks(g1, g2, truster, f1, f2, f3)
        c1, c2, c3 = all_coefs
        coef1.append(c1)
        coef2.append(c2)
        coef3.append(c3)
        print(truster)
    return coef1, coef2, coef3
