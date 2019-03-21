def parse_line(line):
    """
    parses the input line from the sequoia dataset
    ------------------------------------------------------------------
    returns:
            sentence: the input raw line transformed to a sentence

               vocab: a list of tuple (word, pos) associated to the vocabulary of the given sentence,
        unwrap_rules: a list of list containing the rules associated to the sentence
    """
    prev = 0
    sentence = ''
    read_pos = False
    pos = ''
    vocab = []
    rules = []
    level = 0
    for i in range(len(line)):
        c = line[i]
        if read_pos:
            if c == ' ':
                read_pos = False
                pos = pos.split('-')[0]
                if level - 2 >= 0:
                    rules[level - 2][-1].append(pos)
                rules[level - 1][-1].append(pos)
            else:
                pos += c
        if c == '(':
            # in that case, we start reading a symbol
            level += 1
            if len(rules) <= level - 1:
                rules.append([])
            rules[level - 1].append([])
            read_pos = True
            pos = ''
        if c == ')':
            # in that case, we just read a terminal symbol
            level -= 1
            if line[i - 1] != ')':
                word = line[prev + 1:i]
                sentence += word + ' '
                pos = pos.split('-')[0]
                rules[level][-1].append(pos.lower())
                vocab.append((word, pos))
        if c == ' ':
            prev = i
    sentence = sentence[:-1]
    unwrap_rules = []
    for level in range(len(rules)):
        for i in range(len(rules[level])):
            unwrap_rules.append(rules[level][i])
    return sentence, vocab, unwrap_rules[1:]


def create_pcfg(train_set):
    """
    creates a PCFG from the input training set
    ------------------------------------------------------------------------------------------
    Returns:
        heads:
            the list of non terminal symbols
        rules:
            for each symbol in `heads` at index i, `rules[i]` is the list of rules in the CFG
            associated to the head
        freqs_pos:
            for each symbol in `heads` at index i, `freqs_pos[i]` is the list of probabilities
            associated to `rules[i]` in the PCFG
        words:
            for each symbol in `heads` at index i, `words[i]` is the list of terminal symbols
            associated to the PoS (since all symbols in heads are not necessarely PoS, this
            list may be empty)
        freqs_word:
            the frequency associated to the words in the lexicon `word`
    """
    rule_counts = {}
    sentences = []
    rule_string = {}
    lexicon = {}
    for line in train_set:
        # parses every line in the training set, then fill the frequency dictionaries in order
        # to count the occurences of the rules contained in the lines, and the frequencies of
        # each word in the obtained lexicon
        sentence, vocab, rules = parse_line(line)
        sentences.append(sentence)
        for tup in vocab:
            word, pos = tup
            if pos not in lexicon:
                lexicon[pos] = {}
            if word not in lexicon[pos]:
                lexicon[pos][word] = 0
            lexicon[pos][word] += 1
        for rule in rules:
            head = rule[0]
            rule = rule[1:]
            if head not in rule_counts:
                rule_counts[head] = {}
            rule_str = '_'.join(rule)
            if rule_str not in rule_counts[head]:
                rule_counts[head][rule_str] = 0
            rule_counts[head][rule_str] += 1

    # from the computed counts, compute the frequencies of the rules
    heads = []
    rules = []
    freqs_pos = []
    for i, head in enumerate(rule_counts.keys()):
        heads.append(head)
        rules.append([])
        freqs_pos.append([])
        s = sum(rule_counts[head].values())
        for j, rule_str in enumerate(rule_counts[head].keys()):
            rule = rule_str.split('_')
            rules[i].append(rule)
            freqs_pos[i].append(rule_counts[head][rule_str] / s)

    # from the computed counts, compute frequencies associated to the lexicon
    words = [[] for _ in range(len(heads))]
    freqs_word = [[] for _ in range(len(heads))]
    for pos in lexicon.keys():
        i = heads.index(pos)
        s = sum(lexicon[pos].values())
        for j, word in enumerate(lexicon[pos].keys()):
            words[i].append(word)
            freqs_word[i].append(lexicon[pos][word] / s)
    return heads, rules, freqs_pos, words, freqs_word, sentences


def grammar2rulelist(grammar):
    heads, rules, probs = grammar
    rule_list = []
    for i in range(len(heads)):
        head = heads[i]
        rules_h, probs_h = rules[i], probs[i]
        for j in range(len(rules_h)):
            rule, prob = rules_h[j], probs_h[j]
            rule_list.append((head, rule, prob))

    return rule_list


def chomsky_normal_form(heads, rules, probs):
    heads_ind = [0 for _ in range(len(heads))]
    rule_list = []

    # START & TERM conditions should already be respected
    # BIN
    for i in range(len(heads)):
        head, rules_h, probs_h = heads[i], rules[i], probs[i]
        for j in range(len(rules_h)):
            rule, prob = rules_h[j], probs_h[j]

            if len(rule) > 2:
                new_head = head
                ind = heads.index(head)
                for k in range(len(rule)-1):
                    v1 = rule[k]
                    new_prob = prob if k == 0 else 1
                    if k == len(rule)-2:
                        v2 = rule[k+1]
                    else:
                        v2 = head+'_{}'.format(heads_ind[ind])
                        heads_ind[ind] += 1
                    rule_list.append((new_head, [v1, v2], new_prob))
                    new_head = v2
            else:
                rule_list.append((head, rule, prob))

    # UNIT
    unit_rules_id = []
    # create a stack of unit rules id
    for i in range(len(rule_list)):
        head, rule, prob = rule_list[i]
        if len(rule) == 1 and rule[0].upper() == rule[0]:
            unit_rules_id.append(i)

    removed = []
    unit_rules = []
    while not len(unit_rules_id) == 0:
        i = unit_rules_id.pop()
        head, rule, prob = rule_list[i]
        rule_list[i] = None
        symbol = head.split('#')[0]
        removed.append((symbol, rule))
        # add the removed unit rule to the list of unit rules and keep its index
        unit_rules.append((head, rule, prob))
        id_unit = len(unit_rules)-1
        new_head = head+'#{}'.format(id_unit)
        new_rules = []
        for j in range(len(rule_list)):
            if rule_list[j] is None:
                continue
            head_, rule_, prob_ = rule_list[j]
            if head_.split('#')[0] == rule[0]:
                if len(rule_) == 1:
                    if (symbol, rule_) in removed:
                        continue
                    elif rule_[0].upper() == rule_[0]:
                        unit_rules_id.append(len(rule_list) + len(new_rules))
                new_rules.append((new_head, rule_, prob * prob_))
        rule_list = rule_list + new_rules
    # remove from the rule list all the rules that have been set to None
    new_heads = []
    new_rule_list = []
    for i in range(len(rule_list)):
        if rule_list[i] is not None:
            new_rule_list.append(rule_list[i])
            head, _, _ = rule_list[i]
            if head not in new_heads:
                new_heads.append(head)

    rule_list, heads = refined_rule_list(new_rule_list)

    return rule_list, heads


def refined_rule_list(rule_list):
    new_rule_list = []
    true_heads = []
    for i in range(len(rule_list)):
        head, _, _ = rule_list[i]
        head = head.split('#')[0]
        if head not in true_heads:
            true_heads.append(head)

    true_rule_list = {}
    for i in range(len(rule_list)):
        head, rule, prob = rule_list[i]
        if len(head.split('#')) == 1:
            new_rule_list.append((head, rule, prob))
        else:
            true_head = head.split('#')[0]
            rule_str = true_head+'-'+'-'.join(rule)
            if rule_str not in true_rule_list:
                true_rule_list[rule_str] = []
            probs = true_rule_list[rule_str]
            probs.append(prob)
            true_rule_list[rule_str] = probs

    for rule_str in true_rule_list.keys():
        probs = true_rule_list[rule_str]
        total_prob = sum(probs)
        tab = rule_str.split('-')
        head = tab[0]
        rule = tab[1:]
        new_rule_list.append((head, rule, total_prob))
    return new_rule_list, true_heads
