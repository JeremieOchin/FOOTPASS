import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def compute_distance_matrix_batch(avg_positions):
    """
    Computes the pairwise distance matrix for each batch based on average positions using torch.cdist.
    
    Parameters:
    - avg_positions: tensor of shape (B, 22, 2) containing average positions.
    
    Returns:
    - distance_matrices: tensor of shape (B,22,22).
    """

    distance_matrix = torch.cdist(avg_positions, avg_positions, p=2)  # p=2 for Euclidean distance
    

    return distance_matrix


def create_edge_index_batch(distance_matrix, N, d=2):
    """
    Build directed edge indices for a batch from pairwise distances by connecting each node
    to up to N closest neighbors whose distance <= d.

    Args:
        distance_matrix: (B, 26, 26) float tensor of pairwise distances.
        N: int, max #neighbors per node (within threshold).
        d: float, hard distance threshold.

    Returns:
        List[Tensor]: per-graph edge_index tensors of shape (2, E_b), dtype long
    """
    assert distance_matrix.dim() == 3 and distance_matrix.size(1) == distance_matrix.size(2), "distance_matrix must be (B, V, V)"
    
    B, V, _ = distance_matrix.shape
    device = distance_matrix.device
    edge_indices = []
    inf = float('inf')

    for b in range(B):

        dist = distance_matrix[b].clone()
        idx = torch.arange(V, device=device)
        dist[idx, idx] = inf
        dist[dist > d] = inf

        # For each node i, pick up to N nearest finite neighbors
        order = torch.argsort(dist, dim=1)  # (V, V), ascending; +inf goes to the end
        edges = []
        for i in range(V):
            cand = order[i]
            valid = cand[torch.isfinite(dist[i, cand])]  # neighbors within threshold
            nbrs = valid[:N]
            for j in nbrs.tolist():
                edges.append([i, j])

        if edges:
            e = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        else:
            e = torch.empty(2, 0, dtype=torch.long, device=device)

        edge_indices.append(e)

    return edge_indices


def create_edge_attr_batch(data, edge_indices):
    """
    Creates the edge attributes for each batch.
    
    Parameters:
    - data: PyTorch tensor of shape (B, 22, T, 6)
    - edge_indices: List of edge index tensors, where each tensor is of shape (2, E_b)
    - t : timestep
    
    Returns:
    - edge_attrs: List of edge attribute tensors, where each tensor is of shape (T, E_b, 6)
    """
    B,_,T,_ = data.shape
    edge_attrs = []
    
    for b in range(B):

        edges = edge_indices[b]
        batch_edge_attr = []
        
        for src, dst in edges.t():
            delta_pos_vel = data[b, dst, :, 2:6] - data[b, src, :, 2:6]  # (T,4) avec 4 : Xb - Xa, Yb - Ya, Vxb - Vxa, Vyb - Vya 
            dist = torch.sqrt(torch.square(delta_pos_vel[:,:2]).sum(dim=1, keepdim=True)) # (T,1) 
            team_similarity = 1 if data[b, dst, 0, 0] == data[b, src, 0, 0] else 0
            batch_edge_attr.append(torch.cat([delta_pos_vel, dist, torch.tensor([team_similarity]).unsqueeze(0).expand(T,1)], dim=1))
        
        edge_attrs.append(torch.stack(batch_edge_attr, dim=1)) # (T,E_b,6)
    
    return edge_attrs # List of B tensors (T,E_b,6)


def create_time_edge_index_batch(data, M) :

    """
    Creates the edge index by connecting each node to the N closest nodes for each batch.
        
    Parameters:
    - data: PyTorch tensor of shape (B, 22, T, 6)
    - M: PyTorch tensor of shape (B, 1)
        
        Returns:
        - edge_indices: List of edge index tensors, where each tensor is of shape (2, E_b) 
        for batch b.
    """

    B,_,T,_ = data.shape
    
    edge_indices = []

    for b in range(B):

        edges = []
        m = M[b].item()

        for i in range(m):

            for t in range(T-1):

                edges.append([i+t*22, i+(t+1)*22])
                edges.append([i+(t+1)*22, i+t*22])

        edge_indices.append(torch.tensor(edges).t())  # Transpose to shape (2, E_b)
    
    return edge_indices


def create_batched_graphs(data, N):
    """
    Creates batched PyTorch Geometric graphs for all batches and all timesteps.
    
    Parameters:
    - data: PyTorch tensor of shape (B, 22, T, 6) where B is batch size [tmid, shnb, x, y, vx, vy]
    - M: PyTorch tensor of shape (B, 1), indicating the number of valid nodes for each batch
    - N: Number of closest neighbors to connect for each node
    
    Returns:
    - batched_graphs: A single batched PyTorch Geometric Data object containing all graphs
    """
    B, _, T, _ = data.size()  # B is batch size, 22 nodes, T timesteps, 6 features

    # 1. Compute average positions and distance matrix for each batch
    avg_positions = data.clone()
    avg_positions = data[:,:,:,2:4].mean(dim=2)
    distance_matrices = compute_distance_matrix_batch(avg_positions)
    
    # 2. Create edge indices based on N closest neighbors for each batch
    edge_indices = create_edge_index_batch(distance_matrices, N) # List((2, E_b))
    # time_edge_indices = create_time_edge_index_batch(data, M) # List((2, E_b)) #### DISABLED : SWITCHING TOWARD STGNN WITH TEMPORAL CONV / GNN / TEMP CONV etc.
    
    # 3. Create edge attributes for each batch
    edge_attrs = create_edge_attr_batch(data, edge_indices) # List of B tensors (T,E_b,6)

    all_graphs = []
    
    # Create graphs for each timestep across all batches
    for b in range(B):
        
        sub_batched_graph = []
        for t in range(T):

            x = data[b,:,t,[0,2,3,4,5]].clone().half()
            edge_index = edge_indices[b]
            edge_attr = edge_attrs[b][t,:,:]
            
            # Convert to PyTorch Geometric Data object
            sub_batched_graph.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        sub_batched_graph = Batch.from_data_list(sub_batched_graph)
        # sub_batched_graph.edge_index = torch.cat([sub_batched_graph.edge_index, time_edge_indices[b]], dim=-1) #### DISABLED : SWITCHING TOWARD STGNN WITH TEMPORAL CONV / GNN / TEMP CONV etc.

        all_graphs.append(sub_batched_graph)

    # Batch all graphs across both timesteps and batch dimension
    batched_graphs = Batch.from_data_list(all_graphs)
    
    return batched_graphs