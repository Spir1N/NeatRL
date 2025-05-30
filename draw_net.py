import warnings
import graphviz

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, 
             prune_unused=False, node_colors=None, fmt='svg'):
    if graphviz is None:
        warnings.warn("Graphviz not available.")
        return

    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}
    if node_colors is None:
        node_colors = {}

    dot = graphviz.Digraph(format=fmt,
                           node_attr={
                               'shape': 'circle',
                               'fontsize': '12',
                               'height': '0.5',
                               'width': '0.5',
                               'fixedsize': 'true'
                           })

    # Горизонтальная ориентация слева направо
    dot.attr(rankdir='LR')  # Left to Right
    dot.attr(splines='true')
    dot.attr(nodesep='1.0')  # расстояние между узлами
    dot.attr(ranksep='1.2')  # расстояние между слоями

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    used_nodes = set(genome.nodes.keys())

    # Входные узлы
    with dot.subgraph() as s:
        s.attr(rank='same')
        for k in inputs:
            name = node_names.get(k, str(k))
            s.node(name, _attributes={'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'mediumpurple')})

    # Выходные узлы
    with dot.subgraph() as s:
        s.attr(rank='same')
        for k in outputs:
            name = node_names.get(k, str(k))
            s.node(name, _attributes={'style': 'filled', 'fillcolor': node_colors.get(k, 'coral')})

    # Скрытые узлы
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        dot.node(str(n), _attributes={'style': 'filled', 'fillcolor': node_colors.get(n, 'turquoise')})

    # Связи между узлами
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_node, output_node = cg.key
            a = node_names.get(input_node, str(input_node))
            b = node_names.get(output_node, str(output_node))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.2 + abs(cg.weight / 2.0))  # толщина в зависимости от веса
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot
