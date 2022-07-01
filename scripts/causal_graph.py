from typing import List, Tuple
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas, from_pandas_lasso
from networkx.drawing.nx_pydot import write_dot
from IPython.display import Image

import warnings
warnings.filterwarnings("ignore")  # silence warnings


class CausalGraph:
    def __init__(self, data) -> None:
        self.struct_data = data
        self.sm = None
        self.graph_fig = None

    def learn_graph(
            self,
            beta=None,
            max_iter: int = 100,
            h_tol: float = 1e-8,
            w_threshold: float = 0,
            tabu_edges: List[Tuple[str, str]] = None,
            tabu_parent_nodes: List[str] = None,
            tabu_child_nodes: List[str] = None):
        if beta:
            self.sm = from_pandas_lasso(
                X=self.struct_data,
                beta=beta,
                max_iter=max_iter,
                h_tol=h_tol,
                w_threshold=w_threshold,
                tabu_edges=tabu_edges,
                tabu_parent_nodes=tabu_parent_nodes,
                tabu_child_nodes=tabu_child_nodes,)
        else:
            self.sm = from_pandas(
                X=self.struct_data,
                max_iter=max_iter,
                h_tol=h_tol,
                w_threshold=w_threshold,
                tabu_edges=tabu_edges,
                tabu_parent_nodes=tabu_parent_nodes,
                tabu_child_nodes=tabu_child_nodes,)

    def show_graph(
            self,
            size: float = 4.0,
            scale: float = 1.0,
            prog: str = "dot",
            save: bool = False):
        viz = plot_structure(
            self.sm,
            graph_attributes={
                "scale": str(scale), 
                "size": str(size)
                },
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK,
            prog=prog,
        )
        if save:
            viz.draw(path='reports/causal_graph.png')
        return Image(viz.draw(format='png'))

    def remove_edges_below_threshold(self, threshold: float = 0.0):
        self.sm.remove_edges_below_threshold(threshold)

    def save_graph(self, file_name="graph.dot"):
        write_dot(self.sm, f'../models/{file_name}')

    @staticmethod
    def jaccard_similarity(graph1, graph2):
        i = set(graph1).intersection(graph2)
        return round(len(i) / (len(graph1) + len(graph2) - len(i)),3)
