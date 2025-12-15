import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotTransformer(nn.Module):
    def __init__(
        self,
        num_propositions,
        embed_dim,
        num_heads,
        num_layers
    ):
        super().__init__()
        
        # 1. Identity Embeddings: A unique ID vector for every possible fact
        # We assume 'num_propositions' is the total size of our universe (N)
        self.identify_embeddings = nn.Parameter(torch.randn(num_propositions, embed_dim))
        
        # 2. Input Projection: We take [Truth_Score | Identity] -> Transformer_Dim
        # Truth score is 1 dim, Identity is 'embed_dim'. Total_input = embed_dim + 1
        self.input_proj = nn.Linear(embed_dim + 1, embed_dim)
        
        # 3. The Transformer: "Set-to-Set" processing
        # batch_first=True means input is (Batch, N_Slots, Dim)
        # norm_first=True is generally more stable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. The Truth Update Head
        # Projects the processed vector back down to a single probability score
        self.output_head = nn.Linear(embed_dim, 1)
        
    def forward(self, current_truth_scores):
        """
        current_truth_scores: Tensor of shape (Batch_Size, Num_Propositions, 1)
                              Values are between 0 and 1
        """
        batch_size, num_props, _ = current_truth_scores.shape
        
        # Step A: Expand the Identity Embeddings to match the batch size
        # Shape becomes (Batch_Size, Num_Propositions, Embed_Dim)
        identities = self.identify_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Steo B: Create the Slot Representation
        # Concatenate [Truth_Score (1) | Identity (d)]
        # Result shape: (Batch_Size, Num_Propositions, Embed_Dim + 1)
        x = torch.cat([current_truth_scores, identities], dim=-1)
        
        # Step C: Project to correct dimension and run Transformer
        x = self.input_proj(x)
        x = self.transformer(x) # Self-attention magic happens here
        
        # Step D: Predict new truth scores
        logits = self.output_head(x)
        new_truth_scores = torch.sigmoid(logits)
        
        return new_truth_scores