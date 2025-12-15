import torch
import random

random.seed(9001)

def generate_family_batch(batch_size, num_people=5):
    """
    Generates a batch of logical closure tasks.
    
    Returns:
        inputs: (Batch, N, 1) - The 'Partial Knowledge' (facts we tell the AI)
        targets: (Batch, N, 1) - The 'Full Truth' (facts + all entailments)
    """
    
    # We define our universe of propositions.
    # For N people, we have N*N possible 'Parent' relations and N*N 'Ancestor' relations.
    # Total Slots = 2 * N * N
    
    inputs_batch = []
    targets_batch = []
    
    for _ in range(batch_size):
        # 1. Create a random valid family tree (Linear chain for simplicity: A->B->C...)
        # A matrix where edges[i][j] = 1 means i is parent of j
        parent_matrix = torch.zeros(num_people, num_people)
        
        # Randomly assign parents (ensure no cycles for simplicity)
        for i in range(num_people - 1):
            if random.random() > 0.5:
                parent_matrix[i, i+1] = 1 # i is parent of i+1
                

                
        # 2. Compute the Ground Truth (Closure)
        # Ancestor matrix starts as a copy of Parent matrix
        ancestor_matrix = parent_matrix.clone()
        
        # Floyd-Warshall algorithm to compute transitive closure (Logic solver)
        # If i is ancestor of k, and k is ancestor of j, then i is ancestor of j
        for k in range(num_people):
            for i in range(num_people):
                for j in range(num_people):
                    if ancestor_matrix[i, k] and ancestor_matrix[k, j]:
                        ancestor_matrix[i, j] = 1
                        
        # 3. Create Vectors
        # Flatten matrices into 1D vectors
        # Slots 0 to N^2-1 are "ParentOf"
        # Slots N^2 to 2N^2-1 are "AncestorOf"
        flat_parent = parent_matrix.view(-1)
        flat_ancestor = ancestor_matrix.view(-1)
        
        full_truth = torch.cat([flat_parent, flat_ancestor]).unsqueeze(1) # Shape: [2*N*N, 1]
        
        # 4. Create Partial Input
        # We give the model the Parent facts, but HIDE the Ancestor facts.
        # It must derive the Ancestors.
        partial_input = full_truth.clone()
        
        # Mask out the 'Ancestor' section in the input (set to 0.5 for 'unknown' or 0 for false)
        midpoint = len(flat_parent)
        partial_input[midpoint:] = 0.0
        
        inputs_batch.append(partial_input)
        targets_batch.append(full_truth)
        
    return torch.stack(inputs_batch), torch.stack(targets_batch)