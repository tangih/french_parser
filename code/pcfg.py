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


def chomsky_normal_form(heads, rules, probs):
    heads_ind = [0 for _ in range(len(heads))]
    rule_list = []
    for i in range(len(heads)):
        head, rules_h, probs_h = heads[i], rules[i], probs[i]
        for j in range(len(rules_h)):
            rule, prob = rules_h[j], probs_h[j]
            # START & TERM conditions should already be respected
            # BIN
            if len(rule) > 2:
                new_head = head
                for k in range(len(rule)-1):
                    v1 = rule[k]
                    new_prob = prob if k == 0 else 1
                    if k == len(rule)-1:
                        v2 = rule[k+1]
                    else:
                        ind = heads.index(rule[k+1])
                        v2 = rule[k+1]+'_{}'.format(heads_ind[ind])
                        heads_ind[ind] += 1
                    rule_list.append((new_head, [v1, v2], new_prob))
                    new_head = v2
            else:
                rule_list.append((head, rule, prob))
    # UNIT
    unit_rules_id = []
    for i in range(len(rule_list)):
        head, rule, prob = rule_list[i]
        symbol = rule[0]
        if len(rule) == 1 and symbol.upper() == symbol:
            # symbol.upper() == symbol[0] is True iff the symbol is non-terminal
            unit_rules_id.append(i)
    while not len(unit_rules_id) == 0:
        i = unit_rules_id.pop()
        head, rule, prob = rule_list[i]
        symbol = rule[0]
        rule_list[i] = None

        for j in range(len(rule_list)):
            if rule_list[i] is None:
                continue
            head_, rule_, prob_ = rule_list[i]
            if head_ == symbol:
                rule_list.append((head, rule_, prob * prob_))
                if len(rule_) == 1 and rule_[0].upper() == rule_[0]:
                    # symbol.upper() == symbol[0] is True iff the symbol is non-terminal
                    unit_rules_id.append(len(rule_list)-1)

    # dismiss all entries that have been set to None
    new_rule_list = []
    for i in range(len(rule_list)):
        if rule_list[i] is not None:
            new_rule_list.append(rule_list[i])

    # reformat the PCFG
    new_heads = []
    new_rules = []
    new_probs = []
    for i in range(len(new_rule_list)):
        head, rule, prob = new_rule_list[i]
        if head not in new_heads:
            new_heads.append(head)
            new_rules.append([])
            new_probs.append([])
            head_ind = len(new_heads)-1
        else:
            head_ind = new_heads.index(head)
        new_rules[head_ind].append(rule)
        new_probs[head_ind].append(prob)

    return new_heads, new_rules, new_probs
