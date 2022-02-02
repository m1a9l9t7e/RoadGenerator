import os.path

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# from fibre_scenes import Forest


def get_positions(graph_dict, layout_type='neato', name='graph.png', draw=False):
    graph = nx.DiGraph()

    for v in graph_dict.keys():
        graph.add_node(v)

    for delta in graph_dict.items():
        for w in delta[1]:
            graph.add_edge(delta[0],w)

    pos = nx.nx_agraph.graphviz_layout(graph, prog=layout_type)

    if draw:
        options = {
            'node_color': 'orange',
            'edge_color': 'green',
            'node_size': 10,
            'width': 1,
            'alpha': 0.9,
            'with_labels': True,
            'font_weight': 'bold',
            'font_size': 3,
            'arrows': False
        }
        fig = plt.figure()
        nx.draw(graph, pos, **options)
        out_dir = os.path.join(os.getcwd(), 'matplot')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, name), dpi=700)

    return pos


if __name__ == '__main__':
    # forest = Forest(path_to_csv='/home/malte/svenja/gfibre/Export_CableV5.csv')
    forest = None
    graph_dict = forest.to_dict()
    
    # layout_types = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'nop', 'gc', 'acyclic', 'gvpr', 'gvcolor', 'ccomps', 'sccmap', 'tred', 'sfdp', 'unflatten']
    layout_types = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']
    for layout_type in tqdm(layout_types, desc='rendering layouts'):
        positions = get_positions(graph_dict, layout_type=layout_type, name='{}.png'.format(layout_type), draw=True)
