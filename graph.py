from typing import List, Optional

import networkx as nx
import numpy as np

from saver import BaseAdapter

# random_state = np.random.RandomState(42)


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


def extended_barabasi_albert_digraph(
    n0: int,
    n: int,
    m: int,
    p: float,
    q: float,
    nb_attacker: int,
    list_links: list[int],
    seed: Optional[np.random.RandomState] = None,
    vilain_id: Optional[int] = None,
):
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
        seed = np.random.RandomState(42)

    if not (n0 >= 1 and m >= 1 and m < n and n0 < n):
        raise ValueError("Invalid values for n0, n, or m.")
    if not (0 <= p + q < 1):
        raise ValueError("Probabilities p and q must satisfy p + q < 1.")

    # Start with n0 initial nodes
    G: nx.DiGraph = nx.DiGraph()
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
                    src_node, dst_node = sd[seed.randint(0, len(sd))]
                    new_dst_node = seed.choice(attachment_preference)
                    if new_dst_node not in [src_node, dst_node] and not G.has_edge(
                        src_node, new_dst_node
                    ):
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
    relayer = G.number_of_nodes()
    vilain_node = G.number_of_nodes() + 1 if vilain_id is None else vilain_id

    for node in list_links:
        G.add_edge(node, vilain_node)

    bots = []
    bot = G.number_of_nodes() + 1

    while bot < 2 * relayer or bot > n + nb_attacker:
        bots.append(bot)
        G.add_edge(relayer, bot)
        bot = G.number_of_nodes()

    for bot in bots:
        G.add_edge(bot, vilain_node)

    nx.set_edge_attributes(G, 1.0, "capacity")

    return G, vilain_node, relayer, bots


def add_bad_guys(
    G: nx.DiGraph, victim_node: int, relayer: int
) -> tuple[nx.DiGraph, int, int, List[int]]:
    G.add_edge(victim_node, relayer)
    nx.set_edge_attributes(G, 1.0, "capacity")


class GraphAdapter(BaseAdapter[tuple[nx.DiGraph, int, int]]):
    @staticmethod
    def save(full: tuple[nx.DiGraph, int, int]) -> list[str]:
        obj, vilain, relayer = full
        res: list[str] = []
        res.append(
            f"{obj.number_of_nodes()} {obj.number_of_edges()} {vilain} {relayer}"
        )
        for node in obj.nodes:
            new_line = f"{node} {obj.out_degree(node)}"
            for nex in obj.successors(node):
                new_line += f" {nex}"
            res.append(new_line)
        return res

    @staticmethod
    def load(lines: list[str]) -> tuple[nx.DiGraph, int, int]:
        res = nx.DiGraph()
        info_temp = lines.pop(0)
        nb_node, nb_edge, vilain, relayer = [int(val) for val in info_temp.split()]

        for _ in range(nb_node):
            res.add_node(1)

        for node_line in lines:
            infos = node_line.split()
            node_number = int(infos.pop(0))
            nb_successor = int(infos.pop(0))
            successors = [int(val) for val in infos]
            for suc in successors:
                res.add_edge(node_number, suc)
            if nb_successor != len(list(res.successors(node_number))):
                raise Exception("The number of successor isn't good")

        if nb_node != res.number_of_nodes():
            raise Exception("The number of node isn't good")
        if nb_edge != res.number_of_edges():
            raise Exception("The number of edges isn't good")
        nx.set_edge_attributes(res, 1.0, "capacity")

        return res, vilain, relayer
