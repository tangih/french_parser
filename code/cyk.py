import numpy as np


def cyk(sentence_pos, rule_list, heads):
    head_index = {head: index for index, head in enumerate(heads)}
    n = len(sentence_pos)
    r = len(heads)
    P = np.ones((n, n, r), dtype=np.float)
    back = [[[(-1, -1, -1) for _ in range(r)] for _ in range(n)] for _ in range(n)]

    for s in range(n):
        for tup in rule_list:
            head, rule, prob = tup
            if len(rule) == 1 and rule[0] == sentence_pos[s]:
                v = head_index[head]
                P[0, s, v] = prob  # for precision issues take log-prob
                if head[:2] == 'NP':
                    print(head, rule, prob)

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

                    if head[:4] == 'SENT' and rule[0][:4] == 'PONC' and rule[1][:2] == 'NP':
                        if P[0, 1, c] != 1.:
                            print(head, rule)
                            print(P[0, 0, b], P[0, 1, c])
                    p_s = prob * P[p-1, s-1, b] * P[l-p-1, s+p-2, c]

                    if P[p-1, s-1, b] > 0 and P[l - p - 1, s + p - 2, c] > 0 and P[l - 2, s-1, a] < p_s:
                        P[l-1, s-1, a] = p_s
                        back[l-1][s-1][a] = (p, b, c)

    print(P[1, 0, 0])


def decode(back, heads, n):
    assert back[n, 0, 0] != (-1, -1, -1), 'Sentence cannot be parsed with input grammar'
    queue = [back[n-1, 0, 0]]
    while not len(queue) == 0:
        p, b, c = queue.pop()

