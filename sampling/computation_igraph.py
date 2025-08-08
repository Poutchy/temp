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

    # rank = trust_graph.distances(source=truster, target=trustee, algorithm="dijkstra")
    rank = len(trust_graph.get_shortest_path(truster, trustee, algorithm="dijkstra"))
    # except ig:
    #     rank = -1  # float('inf')  # No path found
    return rank


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
    if trustee in trust_graph.neighbors(truster, mode="out"):
        # Direct trust value exists
        # return 100 #trust_graph[truster][trustee]['weight']

        return 100
    # Indirect computation

    score = 0
    neighbor_of_trustee = trust_graph.neighbors(trustee, mode="in")
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
    if trustee in trust_graph.neighbors(truster, mode="out"):
        # Direct trust value exists
        # return 100 #trust_graph[truster][trustee]['weight']

        return 100
    # Indirect computation

    score = 0
    outgoing_of_truster = trust_graph.neighbors(truster, mode="out")
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

        successors = list(graph.neighbors(node, mode="out"))
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
