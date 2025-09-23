import networkx as nx
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from utils.parcorr import ParCorr
from tqdm import tqdm
from sklearn.decomposition import PCA

class ExtendedSummaryGraph:
    """Graph structure for the extended (past/present) representation."""
    def __init__(self, nodes):
        self.nodes_present, self.nodes_past, self.map_names_nodes = self._get_nodes(nodes)
        self.ghat = nx.DiGraph()
        self.ghat.add_nodes_from(self.nodes_present + self.nodes_past)

        # Start with a fully connected graph based on temporal constraints
        for node_present in self.nodes_present:
            for node_past in self.nodes_past:
                self.ghat.add_edge(node_past, node_present)
            for node_present_2 in self.nodes_present:
                if node_present != node_present_2:
                    self.ghat.add_edge(node_present_2, node_present)
                    self.ghat.add_edge(node_present, node_present_2)
        
        self.d = len(nodes) * 2
        self.sep = {edge: [] for edge in self.ghat.edges}

    @staticmethod
    def _get_nodes(names):
        nodes_present, nodes_past, map_names_nodes = [], [], {}
        for name in names:
            node_present = f"{name}_t"
            node_past = f"{name}_t-"
            nodes_present.append(node_present)
            nodes_past.append(node_past)
            map_names_nodes[name] = [node_past, node_present]
        return nodes_present, nodes_past, map_names_nodes

    def add_sep(self, node_p, node_q, node_r):
        if self.ghat.has_edge(node_p, node_q):
            self.sep[(node_p, node_q)].append(node_r)
        if self.ghat.has_edge(node_q, node_p):
            self.sep[(node_q, node_p)].append(node_r)

    def to_summary_graph(self):
        """Converts the extended graph to a final summary graph."""
        nodes = list(self.map_names_nodes.keys())
        map_inv = {node_t: node for node, nodes_t in self.map_names_nodes.items() for node_t in nodes_t}
        
        summary_graph = nx.DiGraph()
        summary_graph.add_nodes_from(nodes)
        
        for node_p_t, node_q_t in self.ghat.edges:
            node_p, node_q = map_inv[node_p_t], map_inv[node_q_t]
            if not summary_graph.has_edge(node_p, node_q):
                summary_graph.add_edge(node_p, node_q)
        return summary_graph

def _run_ci_test(x, y, z=None):
    """Runs a conditional independence test using Partial Correlation."""
    cd = ParCorr(significance='analytic')
    dim_x = x.shape[1]
    dim_y = y.shape[1]

    if z is not None and not z.empty:
        dim_z = z.shape[1]
        X = np.concatenate((x.values, y.values, z.values), axis=1).astype('float64')
        xyz = np.array([0] * dim_x + [1] * dim_y + [2] * dim_z)
    else:
        X = np.concatenate((x, y), axis=1).astype('float64')
        xyz = np.array([0] * dim_x + [1] * dim_y)

    value = cd.get_dependence_measure(X.T, xyz)
    p_value = cd.get_analytic_significance(value=value, T=X.shape[0], dim=X.shape[1], xyz=xyz)
    return p_value

class PCGCE:
    """PC algorithm for Granger Causality Extraction."""
    def __init__(self, series, sig_level=0.05, max_lags=5, verbose=False, num_processor=-1):
        self.series = series.reset_index(drop=True)
        self.n, self.d = self.series.shape
        self.names = self.series.columns
        self.sig_level = sig_level
        self.max_lags = max_lags
        self.verbose = verbose
        self.num_processor = num_processor
        self.graph = ExtendedSummaryGraph(self.names)
        self.data_dict = self._create_data_windows()

        num_nodes = self.graph.ghat.number_of_nodes()
        self.mi_df = pd.DataFrame(np.ones((num_nodes, num_nodes)),
                                  columns=self.graph.ghat.nodes(),
                                  index=self.graph.ghat.nodes())

    def _create_data_windows(self):
        """Creates windowed representations for past and present of each variable."""
        data_dict = {}
        for name in self.names:
            node_past, node_present = self.graph.map_names_nodes[name]
            ts = self.series[name].dropna()
            
            past_windows = pd.DataFrame()
            if self.max_lags > 0:
                for i in range(self.max_lags):
                    past_windows[f'{node_past}_{self.max_lags-i}'] = ts.iloc[i : len(ts)-self.max_lags+i].values
                
                # Use PCA to reduce the past to a single dimension
                pca = PCA(n_components=1)
                past_reduced = pd.DataFrame(pca.fit_transform(past_windows), columns=[node_past])
                
                present = ts.iloc[self.max_lags:].rename(node_present).to_frame().reset_index(drop=True)
                data_dict[node_past] = past_reduced
                data_dict[node_present] = present
        return data_dict

    def _unconditional_test(self, node_p, node_q):
        """Test for unconditional independence between two nodes."""
        x = self.data_dict[node_p]
        y = self.data_dict[node_q]
        min_len = min(len(x), len(y))
        p_val = _run_ci_test(x.head(min_len), y.head(min_len))
        return node_p, node_q, p_val

    def skeleton_initialize(self):
        """Initialize graph skeleton by removing unconditional independencies."""
        if self.verbose: print("--- Phase 1: Skeleton Initialization ---")
        
        edges_to_test = [edge for edge in self.graph.ghat.edges if (edge[1], edge[0]) not in self.graph.ghat.edges or edge[0] < edge[1]]
        
        # Can be parallelized for performance
        results = [self._unconditional_test(p, q) for p, q in tqdm(edges_to_test, desc="Skeleton")]
        
        for node_p, node_q, p_val in results:
            self.mi_df.loc[node_p, node_q] = self.mi_df.loc[node_q, node_p] = p_val
            if p_val > self.sig_level:
                if self.graph.ghat.has_edge(node_p, node_q): self.graph.ghat.remove_edge(node_p, node_q)
                if self.graph.ghat.has_edge(node_q, node_p): self.graph.ghat.remove_edge(node_q, node_p)
                if self.verbose: print(f"Removed edge {node_p}-{node_q} (p-value: {p_val:.4f})")
    
    def find_sep_set(self):
        """Find separation sets for remaining edges using conditional tests."""
        if self.verbose: print("\n--- Phase 2: Finding Separation Sets ---")
        
        set_size = 1
        while True:
            removed_an_edge = False
            edges_to_check = list(self.graph.ghat.edges)
            if self.verbose: print(f"Checking conditioning sets of size {set_size}...")

            for node_p, node_q in tqdm(edges_to_check, desc=f"Sepset size {set_size}"):
                if not self.graph.ghat.has_edge(node_p, node_q): continue
                
                neighbors = set(self.graph.ghat.predecessors(node_p)) | set(self.graph.ghat.successors(node_p))
                neighbors.discard(node_q)
                
                if len(neighbors) < set_size: continue

                for r_set in itertools.combinations(neighbors, set_size):
                    z_data = pd.concat([self.data_dict[r] for r in r_set], axis=1)
                    x_data = self.data_dict[node_p]
                    y_data = self.data_dict[node_q]
                    min_len = min(len(x_data), len(y_data), len(z_data))

                    p_val = _run_ci_test(x_data.head(min_len), y_data.head(min_len), z_data.head(min_len))

                    if p_val > self.sig_level:
                        if self.graph.ghat.has_edge(node_p, node_q): self.graph.ghat.remove_edge(node_p, node_q)
                        if self.graph.ghat.has_edge(node_q, node_p): self.graph.ghat.remove_edge(node_q, node_p)
                        self.graph.add_sep(node_p, node_q, r_set)
                        removed_an_edge = True
                        if self.verbose: print(f"Removed edge {node_p}-{node_q} | {r_set} (p-value: {p_val:.4f})")
                        break # Move to next edge
                if not self.graph.ghat.has_edge(node_p, node_q): continue
            
            set_size += 1
            if not removed_an_edge or set_size > len(self.graph.ghat.nodes()) - 2:
                break
    
    def rule_origin_causality(self):
        """Rule 0 (origin of causality): Orient colliders."""
        if self.verbose: print("\n--- Phase 3: Orienting Edges (Colliders) ---")
        
        for node_r in self.graph.ghat.nodes():
            predecessors = list(self.graph.ghat.predecessors(node_r))
            if len(predecessors) < 2: continue
            
            for node_p, node_q in itertools.combinations(predecessors, 2):
                # Check if p-r-q is a potential collider
                if not self.graph.ghat.has_edge(node_p, node_q) and not self.graph.ghat.has_edge(node_q, node_p):
                    # Check if r is in the separation set of p and q
                    is_in_sep_set = False
                    if (node_p, node_r) in self.graph.sep:
                        if node_q in self.graph.sep[(node_p, node_r)]: is_in_sep_set = True
                    if (node_r, node_p) in self.graph.sep:
                        if node_q in self.graph.sep[(node_r, node_p)]: is_in_sep_set = True
                    
                    if not is_in_sep_set:
                        if self.verbose: print(f"Orienting collider: {node_p} -> {node_r} <- {node_q}")
                        # This is a collider, orient edges if they are bidirectional
                        if self.graph.ghat.has_edge(node_r, node_p): self.graph.ghat.remove_edge(node_r, node_p)
                        if self.graph.ghat.has_edge(node_r, node_q): self.graph.ghat.remove_edge(node_r, node_q)
                        
    def fit(self):
        """Run the full PCGCE algorithm."""
        self.skeleton_initialize()
        self.find_sep_set()
        self.rule_origin_causality()
        
        if self.verbose: print("\n--- PCGCE Finished ---")
        return self.graph.to_summary_graph()