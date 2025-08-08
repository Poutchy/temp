import numpy as np
from matplotlib import pyplot as plt


def plot_single_computation(node_id, scores, min_max_scores, name):
    score_normal, score_attack = scores

    normal, attack = min_max_scores
    normal_min, normal_max = normal
    attack_min, attack_max = attack

    bar_width = 0.4
    x_normal = node_id - bar_width / 2
    x_attack = node_id + bar_width / 2

    plt.figure(figsize=(6, 6))

    # Plot bars
    plt.bar(x_normal, score_normal, width=bar_width, label="Normal", color="skyblue")
    plt.bar(x_attack, score_attack, width=bar_width, label="Attack", color="salmon")

    # Plot min-max lines for normal
    plt.hlines(
        [normal_min, normal_max],
        x_normal - bar_width / 2,
        x_normal + bar_width / 2,
        colors="blue",
        linestyles="dashed",
    )

    # Plot min-max lines for attack
    plt.hlines(
        [attack_min, attack_max],
        x_attack - bar_width / 2,
        x_attack + bar_width / 2,
        colors="red",
        linestyles="dashed",
    )

    # Labeling
    plt.xticks([node_id], [f"Node {node_id}"])
    plt.ylabel("Score")
    plt.title("Normal vs Attack Scores with Min-Max Ranges for the node")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./data/{name}.jpg")
    plt.close()


def plot_multi_computation(node_id, scores_dict, min_max_dict, name):
    bar_width = 0.15
    spacing = bar_width * 3
    x_base = node_id

    plt.figure(figsize=(8, 6))

    colors = {
        "normal": ("skyblue", "salmon"),
        "reverse": ("mediumblue", "indianred"),
        "independent": ("teal", "tomato"),
    }

    for i, (key, (score_normal, score_attack)) in enumerate(scores_dict.items()):
        offset = (i - 1) * spacing  # center around node_id
        x_normal = x_base + offset - bar_width / 2
        x_attack = x_base + offset + bar_width / 2

        color_normal, color_attack = colors[key]

        # Bars
        plt.bar(
            x_normal,
            score_normal,
            width=bar_width,
            label=f"{key} Normal",
            color=color_normal,
        )
        plt.bar(
            x_attack,
            score_attack,
            width=bar_width,
            label=f"{key} Attack",
            color=color_attack,
        )

        # Min/max lines
        (normal_min, normal_max), (attack_min, attack_max) = min_max_dict[key]

        plt.hlines(
            [normal_min, normal_max],
            x_normal - bar_width / 2,
            x_normal + bar_width / 2,
            colors=color_normal,
            linestyles="dashed",
        )
        plt.hlines(
            [attack_min, attack_max],
            x_attack - bar_width / 2,
            x_attack + bar_width / 2,
            colors=color_attack,
            linestyles="dashed",
        )

    # Labels
    plt.xticks([node_id], [f"Node {node_id}"])
    plt.ylabel("Score")
    plt.title("Normal vs Attack Scores (3 Computation Types)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"./data/{name}.jpg")
    plt.close()


def plot_spearman_all(cors):
    # Compute the average Spearman correlation
    cor1, cor2, cor3 = cors
    avg_rho1 = np.mean(cor1)
    avg_rho2 = np.mean(cor2)
    avg_rho3 = np.mean(cor3)
    print(f"Average Spearman correlation {avg_rho1:.4f}")
    print(f"Average Spearman correlation {avg_rho2:.4f}")
    print(f"Average Spearman correlation {avg_rho3:.4f}")

    # Plot the Spearman correlation coefficients
    plt.figure(figsize=(8, 6))
    plt.axhline(
        y=avg_rho1,
        color="r",
        linestyle="--",
        label=f"Average normal (normal/attack) {avg_rho1:.4f}",
    )
    plt.axhline(
        y=avg_rho2,
        color="g",
        linestyle="--",
        label=f"Average (normal/attack) {avg_rho2:.4f}",
    )
    plt.axhline(
        y=avg_rho3,
        color="b",
        linestyle="--",
        label=f"Average (normal/attack) {avg_rho3:.4f}",
    )
    plt.plot(
        cor1,
        marker="o",
        color="red",
        label="Spearman Correlation for a user (normal/attack)",
    )
    plt.plot(
        cor2,
        marker="o",
        color="green",
        label="Spearman Correlation for a user (normal/attack)",
    )
    plt.plot(
        cor3,
        marker="o",
        color="blue",
        label="Spearman Correlation for a user (normal/attack)",
    )
    plt.title("Spearman Correlation Coefficients", fontsize=14)
    plt.xlabel("Node ID", fontsize=12)
    # plt.yscale("log")
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig("./data/comparison.jpg")
    plt.close()


def compare_single_result(normal_graph, attack_graph, score):
    n_points = len(normal_graph)
    bar_width = 0.4
    x = np.arange(n_points)

    plt.figure(figsize=(max(12, n_points * 0.6 / 4), 6))

    # Offset bars slightly
    plt.bar(
        x - bar_width / 2,
        normal_graph,
        width=bar_width,
        color="C0",
        label=f"Normal score",
    )
    plt.bar(
        x + bar_width / 2,
        attack_graph,
        width=bar_width,
        color="C1",
        label=f"Attack score",
    )

    plt.xlabel("For all nodes I taken as victim")
    plt.ylabel("Rank of the vilain as seen by node I")
    plt.title(
        f"Rank evolution with and without an attack for the {score} score calculation"
    )
    plt.xticks(x[::5], [str(i) for i in x[::5]])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./data/{score}.jpg")
    plt.show()


def compare_all_result(normal_graphs, attack_graphs):
    n_points = len(normal_graphs)  # how many x-values
    tuple_size = len(normal_graphs[0])  # how many values per tuple
    total_bars_per_group = tuple_size * 2  # normal + attack per tuple element
    bar_width = 1 / total_bars_per_group
    x = np.arange(n_points)

    plt.figure(figsize=(max(12, n_points * 0.6 / 4), 6))

    for t_idx in range(tuple_size):
        normal_vals = [normal_graphs[i][t_idx] for i in range(n_points)]
        attack_vals = [attack_graphs[i][t_idx] for i in range(n_points)]

        offset = (t_idx * 2 - total_bars_per_group // 2) * bar_width + bar_width / 2

        # Only label once per bar type
        label_normal = (
            "Normal score for the normal score calculation"
            if t_idx == 0
            else (
                "Normal score for the reverse score calculation"
                if t_idx == 1
                else "Normal score for the independent score calculation"
            )
        )
        label_attack = (
            "Attack score for the normal score calculation"
            if t_idx == 0
            else (
                "Attack score for the reverse score calculation"
                if t_idx == 1
                else "Attack score for the independent score calculation"
            )
        )

        plt.bar(
            x + offset,
            normal_vals,
            width=bar_width,
            color=f"C{t_idx}",
            label=label_normal,
        )
        plt.bar(
            x + offset + bar_width,
            attack_vals,
            width=bar_width,
            color=f"C{t_idx}",
            hatch="//",
            label=label_attack,
        )

    plt.xlabel("For all nodes I taken as victim")
    plt.ylabel("Rank of the vilain as seen by node I")
    plt.title("Rank evolution with and without an attack")
    plt.xticks(x[::5], [str(i) for i in x[::5]])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data/all_comparison.jpg")
    plt.show()
