from dataclasses import dataclass, field
from os import makedirs
from typing import Literal, Union

from networkx import DiGraph

from computation import (compute_wot_score, compute_wot_score_indep,
                         compute_wot_score_rev)
from plotting import plot_multi_computation, plot_single_computation
from scipy.stats import rankdata

@dataclass
class SingleScoreComputation:
    node: int
    score: int


@dataclass(init=False)
class ScoreComputation:
    list_normal: list[SingleScoreComputation] = field(default_factory=list)
    list_attack: list[SingleScoreComputation] = field(default_factory=list)

    def __init__(
        self, scores_normal: list[tuple[int, int]], scores_attack: list[tuple[int, int]]
    ) -> None:
        ranks = rankdata([s for _,s in scores_normal])
        self.dict_normal = { n: ranks[i] for i, (n, _) in enumerate(scores_normal)}
        ranks = rankdata([s for _,s in scores_attack])
        self.dict_attack = { n: ranks[i] for i, (n, _) in enumerate(scores_attack)}

    def get_rank(
        self, node: int, mode: Literal["normal", "attack"]
    ) -> int:
        target_dict = self.dict_normal if mode == "normal" else self.dict_attack
        if node in target_dict.keys():
            return target_dict[node]
        raise ValueError("Missing node in the graph")


def group_computations(
    graph_normal: DiGraph, graph_attack: DiGraph, truster: int
) -> tuple[
    tuple[list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
]:
    scores_normal = []
    scores_normal_rev = []
    scores_normal_indep = []
    scores_attack = []
    scores_attack_rev = []
    scores_attack_indep = []

    for trustee in [
        node for node in range(graph_normal.number_of_nodes()) if node < 102
    ]:
        score_normal = compute_wot_score(graph_normal, truster, trustee)
        score_normal_rev = compute_wot_score_rev(graph_normal, truster, trustee)
        score_normal_indep = compute_wot_score_indep(graph_normal, truster, trustee)
        score_attack = compute_wot_score(graph_attack, truster, trustee)
        score_attack_rev = compute_wot_score_rev(graph_attack, truster, trustee)
        score_attack_indep = compute_wot_score_indep(graph_attack, truster, trustee)

        scores_normal.append(score_normal)
        scores_normal_rev.append(score_normal_rev)
        scores_normal_indep.append(score_normal_indep)
        scores_attack.append(score_attack)
        scores_attack_rev.append(score_attack_rev)
        scores_attack_indep.append(score_attack_indep)

    return (
        (scores_normal, scores_normal_rev, scores_normal_indep),
        (scores_attack, scores_attack_rev, scores_attack_indep),
    )


def prepare_datas(graph_normal, graph_attack, truster):
    n, a = group_computations(graph_normal, graph_attack, truster)
    nn, nr, ni = n
    an, ar, ai = a

    return (
        ScoreComputation(list(enumerate(nn)), list(enumerate(an))),
        ScoreComputation(list(enumerate(nr)), list(enumerate(ar))),
        ScoreComputation(list(enumerate(ni)), list(enumerate(ai))),
    )


def prepare_plotting(graph_normal, graph_attack, truster, vilain):
    n, r, i = prepare_datas(graph_normal, graph_attack, truster)

    snn, san = n.get_rank(vilain, "normal"), n.get_rank(vilain, "attack")
    snr, sar = r.get_rank(vilain, "normal"), r.get_rank(vilain, "attack")
    sni, sai = i.get_rank(vilain, "normal"), i.get_rank(vilain, "attack")

    # print(n)
    # print(n.list_normal)
    # print(n.list_normal[-1])
    # print(n.list_normal[-1].node)

    max_nn, min_nn = len(n.list_normal) - 1, 0
    max_nr, min_nr = len(r.list_normal) - 1, 0
    max_ni, min_ni = len(i.list_normal) - 1, 0
    max_an, min_an = len(n.list_attack) - 1, 0
    max_ar, min_ar = len(r.list_attack) - 1, 0
    max_ai, min_ai = len(i.list_attack) - 1, 0

    makedirs(f"./data/{truster}", exist_ok=True)

    plot_single_computation(
        vilain, (snn, san), ((min_nn, max_nn), (min_an, max_an)), f"{truster}/normal"
    )
    plot_single_computation(
        vilain, (snr, sar), ((min_nr, max_nr), (min_ar, max_ar)), f"{truster}/reverse"
    )
    plot_single_computation(
        vilain,
        (sni, sai),
        ((min_ni, max_ni), (min_ai, max_ai)),
        f"{truster}/independent",
    )
    plot_multi_computation(
        vilain,
        {"normal": (snn, san), "reverse": (snr, sar), "independent": (sni, sai)},
        {
            "normal": ((min_nn, max_nn), (min_an, max_an)),
            "reverse": ((min_nr, max_nr), (min_ar, max_ar)),
            "independent": ((min_ni, max_ni), (min_ai, max_ai)),
        },
        f"{truster}/all",
    )

    print(san)

    # print(f"{snn} -> {san}")
    # print(f"{snr} -> {sar}")
    # print(f"{sni} -> {sai}")

    return (snn, snr, sni), (san, sar, sai)
