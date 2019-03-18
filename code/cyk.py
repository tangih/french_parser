import numpy as np


def cyk(sentence_pos, grammar):
    heads, rules, probs = grammar
    head_index = {head: index for index, head in enumerate(heads)}
    print('HEAD INDEX OF SENT SYMBOL: {}'.format(head_index['SENT']))
    n = len(sentence_pos)
    r = len(heads)
    P = np.zeros((n, n, r), dtype=np.float)
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
                print(head, rule, prob)
                v = head_index[head]
                P[0, s, v] = prob
    for l in range(2, n+1):
        for s in range(1, n-l+2):
            for p in range(1, l):
                for tup in rule_list:
                    head, rule, prob = tup
                    if len(rule) != 2:
                        continue
                    a = head_index[head]
                    b = head_index[rule[0]]
                    c = head_index[rule[1]]
                    if head == 'SENT':
                        print(head, rule)
                        print(prob, P[p - 1, s - 1, b], P[l - p - 1, s + p - 2, c])
                    p_s = prob * P[p-1, s-1, b] * P[l-p-1, s+p-2, c]
                    print(p_s)
                    # print(p_s)
                    if P[p, s, b] > 0 and P[l - p - 1, s + p, c] > 0 and P[l - 1, s, a] < p_s:
                        print('#############################################')
                        P[l, s, a] = p_s
                        back[l][s][a] = (p, b, c)
    for i in range(n):
        for j in range(n):
            for k in range(r):
                if back[i][j][k] != (-1, -1, -1):
                    print(i, j, k)
                    print(back[i][j][k])
                    print('###################################')
