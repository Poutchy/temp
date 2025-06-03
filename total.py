import json

import numpy as np

from graph import GraphAdapter, add_bad_guys, extended_barabasi_albert_digraph
from group_data import prepare_plotting
from plotting import compare_all_result, compare_single_result
from saver import load, save

with open("config.json", "r", encoding="utf-8") as jsonfile:
    data = json.load(jsonfile)
    print("Read successful")

k = 2

list_nodes = list(range(data["number_normal_node"]))
np.random.RandomState(42).shuffle(list_nodes)
list_nodes = list_nodes[:2]

g_normal, vilain, relayer, _ = extended_barabasi_albert_digraph(
    data["number_starting_graph"],
    data["number_normal_node"],
    data["number_edge_new_node"],
    data["probability_add_triangle"],
    data["probability_add_random_edge"],
    data["number_normal_node"],
    list_nodes,
)

save((g_normal, vilain, relayer), GraphAdapter, "./data/graph.txt")
g_normal, vilain, relayer = load(GraphAdapter, "./data/graph.txt")
# g_attack, vilain_attack, relayer_attack = load(GraphAdapter, "./data/graph.txt")


graphs = []
for i in range(data["number_normal_node"]):
    g, _, _ = load(GraphAdapter, "./data/graph.txt")
    add_bad_guys(g, i, relayer)
    graphs.append(g)


# print(relayer, relayer_attack, relayer == relayer_attack)
# print(vilain, vilain_attack, vilain == vilain_attack)


# g_attack, _, _, _ = extended_barabasi_albert_digraph(
#     data["number_starting_graph"],
#     data["number_normal_node"],
#     data["number_edge_new_node"],
#     data["probability_add_triangle"],
#     data["probability_add_random_edge"],
#     data["number_normal_node"],
#     list_nodes,
# )

# add_bad_guys(g_attack, 17, relayer)

# for node in g_normal:
#     if vilain in g_normal.successors(node) and node < 101:
#         print(f"node ok: {node}")
# for node in g_attack:
#     if vilain in g_attack.successors(node) and node < 101:
#         print(f"node ok: {node}")

# print(g_normal == g_attack)

# for node in range(g_normal.number_of_nodes()):
#     if 53 in g_normal.successors(node) or 83 in g_attack.successors(node):
#         print(f"soluce:  {node}")

# print(g_normal.number_of_nodes(), g_attack.number_of_nodes(), g_normal.number_of_nodes() == g_attack.number_of_nodes())
# print(g_normal.number_of_edges(), g_attack.number_of_edges(), g_normal.number_of_edges() + 1 == g_attack.number_of_edges())

datas = []

sc_normals = []
sc_attacks = []
for i in range(data["number_normal_node"]):
    sc_normal, sc_attack = prepare_plotting(g_normal, graphs[i], i, vilain)
    sc_normals.append(sc_normal)
    sc_attacks.append(sc_attack)

sc_nn = [normal for normal, _, _ in sc_normals]
sc_an = [normal for normal, _, _ in sc_attacks]
sc_nr = [reverse for _, reverse, _ in sc_normals]
sc_ar = [reverse for _, reverse, _ in sc_attacks]
sc_ni = [indep for _, _, indep in sc_normals]
sc_ai = [indep for _, _, indep in sc_attacks]


compare_single_result(sc_nn, sc_an, "normal")
compare_single_result(sc_nr, sc_ar, "reversed")
compare_single_result(sc_ni, sc_ai, "independent")
compare_all_result(sc_normals, sc_attacks)

# c_all = compute_coef(
#     g_normal,
#     g_attack,
#     compute_wot_score,
#     compute_wot_score_rev,
#     compute_wot_score_indep,
# )
# c_normal, c_reverse, c_independent = c_all
# plot_spearman_all((c_normal, c_reverse, c_independent))
