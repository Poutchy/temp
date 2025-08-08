from numpy import corrcoef


def compare_score_ranks(g1, g2, truster, f1, f2, f3):
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
    fin1 = corrcoef(alls1, alla1)[0, 1]
    fin2 = corrcoef(alls2, alla2)[0, 1]
    fin3 = corrcoef(alls3, alla3)[0, 1]
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