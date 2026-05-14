import json
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd


def build_directed_graph(df: pd.DataFrame, pid_col="sim_pid", contact_col="contact_pid") -> nx.DiGraph:
    """
    Builds a directed graph from infector (contact_pid) to infectee (pid).
    """
    G = nx.DiGraph()
    if pid_col not in df.columns or contact_col not in df.columns:
        return G
    
    pids = df[pid_col].astype(str)
    contacts = df[contact_col].astype(str)
    edges = zip(contacts, pids)
    
    # Valid directed edges (ignore -1, nan, or self-loops)
    valid_edges = [(u, v) for u, v in edges if u and u != "-1" and u != "nan" and u != v]
    G.add_edges_from(valid_edges)
    return G



def simulate_inference_and_matrix(G, known_tip_states, alphabet, simulation_duration_years):
    """
    Simulates phylogenetic inference on the transmission tree to create an 
    inferred geographic transition matrix (mugration format).
    """
    inferred_states = {}
    
    # Traverse from tips up to the root
    try:
        nodes_sorted = list(reversed(list(nx.topological_sort(G))))
    except nx.NetworkXUnfeasible:
        print("Warning: Cycles detected in transmission graph. Falling back to default ordering.")
        nodes_sorted = list(G.nodes())

    for node in nodes_sorted:
        if node in known_tip_states:
            inferred_states[node] = known_tip_states[node]
        else:
            # Look at the children's states to infer this node's state
            children = list(G.successors(node))
            child_states = [inferred_states[c] for c in children if c in inferred_states and inferred_states[c] != ""]
            
            if child_states:
                # Assign the most common child state (most_common returns a list of tuples)
                most_common = Counter(child_states).most_common(1)
                inferred_states[node] = most_common[0][0]
            else:
                inferred_states[node] = ""

    # Count Transitions
    transfer_counts = defaultdict(lambda: defaultdict(int))
    total_jumps = 0
    
    for parent, child in G.edges():
        parent_loc = inferred_states.get(parent, "")
        child_loc = inferred_states.get(child, "")
        
        if parent_loc and child_loc and parent_loc != child_loc:
            transfer_counts[parent_loc][child_loc] += 1
            total_jumps += 1
            
    # Calculate Equilibrium Probabilities
    state_counts = Counter(inferred_states.values())
    total_inferred = sum(state_counts.values())
    
    eq_probs = [0.0] * len(alphabet)
    for i, county in enumerate(alphabet):
        if county != "":
            eq_probs[i] = float(state_counts.get(county, 0) / max(total_inferred, 1))
            
    # Calculate Symmetrized Matrix (W_ij)
    overall_rate = float(total_jumps / simulation_duration_years) if simulation_duration_years > 0 else 0.0
    N = len(alphabet)
    transition_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    
    for i, origin in enumerate(alphabet):
        if origin == "": continue
        for j, dest in enumerate(alphabet):
            if i == j or dest == "": continue
            
            pi_i = eq_probs[i]
            pi_j = eq_probs[j]
            
            if overall_rate == 0 or pi_i == 0 or pi_j == 0:
                continue
            
            # Actual rate = Jumps per year
            q_ij = transfer_counts[origin][dest] / simulation_duration_years
            q_ji = transfer_counts[dest][origin] / simulation_duration_years
            
            # W_ij = Q_ij / (mu * pi_j)
            w_ij = q_ij / (overall_rate * pi_j)
            w_ji = q_ji / (overall_rate * pi_i)
            
            transition_matrix[i][j] = float((w_ij + w_ji) / 2.0)
            
    return {
        "generated_by": {"program": "run_all_scenarios.py", "version": "1.0"},
        "models": {
            "county": {
                "alphabet": alphabet,
                "equilibrium_probabilities": eq_probs,
                "rate": overall_rate,
                "transition_matrix": transition_matrix
            }
        }
    }


def align_and_normalize_matrix(run_matrix, run_alphabet, master_county_names):
    """
    Aligns a sparse transition matrix from Nextstrain to a master list of counties 
    and row-normalizes it so rows sum to 1.
    """
    N = len(master_county_names)
    full_mat = np.zeros((N, N))
    
    # Create a mapping from the run's alphabet index to the master list's index
    index_map = []
    for county in run_alphabet:
        if county in master_county_names:
            index_map.append(master_county_names.index(county))
        else:
            index_map.append(-1) # Ignore unexpected counties
            
    # Map the sparse matrix values to their exact, absolute coordinates in the full matrix
    for i in range(len(run_alphabet)):
        master_i = index_map[i]
        if master_i == -1: 
            continue
            
        for j in range(len(run_alphabet)):
            master_j = index_map[j]
            if master_j == -1: 
                continue
                
            full_mat[master_i, master_j] = run_matrix[i][j]
            
    # Row-normalize the full matrix
    row_sums = full_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for rows with no outbound transfers
    normalized = full_mat / row_sums
    
    return normalized

def read_traits_json(file_path):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    models = data['models']
    
    county_names = models['county']['alphabet']
    transition_matrix = models['county']["transition_matrix"]
    return county_names, transition_matrix

