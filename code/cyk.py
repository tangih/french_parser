import numpy as np


def cyk(sentence_pos, grammar):
    heads, rules, probs = grammar
    head_index = {head: index for index, head in enumerate(heads)}
    n = len(sentence_pos)
    r = len(heads)
    P = np.array((n, n, r), dtype=np.float)
    back = [[[(-1, -1, -1) for _ in range(r)] for _ in range(n)] for _ in range(n)]
    rule_list = []
    for i in range(len(heads)):
        head = heads[i]
        rules_h, probs_h = rules[i], probs[i]
        for j in range(len(rules_h)):
            rule, prob = rules_h[j], probs_h[j]
            rule_list.append((head, rule, prob))
    for s in range(n):
        for tup in rule_list:
            head, rule, prob = tup
            if len(rule) == 1 and rule[0] == sentence_pos[s]:
                v = head_index[head]
                P[0, s, v] = prob
    for l in range(1, n):
        for s in range(n-l+1):
            for p in range(l-1):
                for tup in rule_list:
                    head, rule, prob = tup
                    if len(rule) != 2:
                        continue
                    a = head_index[head]
                    b = head_index[rule[0]]
                    c = head_index[rule[1]]
                    p_s = prob * P[p, s, b] * P[l-p, s+p, c]
                    if P[p, s, b] > 0 and P[l - p, s + p, c] > 0 and P[l, s, a] < p_s:
                        P[l, s, a] = p_s
                        back[l][s][a] = (p, b, c)
