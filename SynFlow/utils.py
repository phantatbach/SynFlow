from collections import defaultdict

def build_graph(tokens, pattern):
    """
    Given a list of conllu tokens and a regex pattern, build a dependency graph,
    a mapping of id to lemma/pos, and a mapping of edge to deprel.

    Returns a tuple of (id2lemma_pos, graph, id2deprel).
    """
    
    id2lemma_pos = {}
    graph        = defaultdict(list)
    id2deprel    = {}

    for tok in tokens:
        m = pattern.match(tok)
        if not m:
            continue
        # You don't need the wordform to create a dependency graph
        _, lemma, pos, idx, head, deprel = m.groups()
        # Create a dictionary of {id: lemma/pos}
        id2lemma_pos[idx] = f'{lemma}/{pos}'
        # If the current word is not the root
        if head != '0':
            # Create an un-directional graph of with 2 dictionary entries {children: [parents]} and {parents: [children]}
            graph[idx].append(head)
            graph[head].append(idx)

            # Create a un-directional graph with 2 dictionary entries {edge: deprel}
            # chi: child-ward, pa: parent-ward
            id2deprel[(idx, head)] = f'pa_{deprel}'
            id2deprel[(head, idx)] = f'chi_{deprel}'
    return id2lemma_pos, graph, id2deprel