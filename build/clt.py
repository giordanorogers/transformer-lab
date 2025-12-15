import torch
import torch.nn as nn
import torch.nn.functional as F

from activation_functions import JumpReLU

class CrossLayerTranscoder(nn.Module):
    """
    A cross-layer transcoder (CLT) where features read from one layer and write to all
    subsequent layers.
    
    Cross-layer transcoders are the core architecture enabling the circuit tracing methodoloy.
    Unlike per-layer transcoders, CLT features can "bridge over" multiple MLP layers, allowing
    a single feature to represent computation that spans the entire forward pass. This dramatically
    shortens paths in attribution graphs by collapsing amplification chains into single features.
    
    Each CLT feature has:
    - One encoder that reads from the residual stream at a specific layer
    - Multiple decoders that can write to all subsequent MLP outputs
    - The ability to represent cross-layer superposition where related computation
    is distributed across multiple transformer layers
    
    A single CLT provides an alternative to using multiple per-layer transcoders (managed by
    TranscoderSet) for feature-based model interpretation and replacement.
    
    Attributes:
        n_layers: Number of transformer layers the CLT spans
        d_transcoders: Number of features per layer
        d_model: Dimension of transformer residual stream
        W_enc: Encoder weights for each layer [n_layers, d_transcoder, d_model]
        W_dec: Decoder weights (lazily loaded) for cross-layer outputs
        b_enc: Encoder biases [n_layers, d_transcoder]
        b_dec: Decoder biases [n_layers, d_model]
        activation_function: Sparsity-inducing nonlinearity (default: ReLU)
        lazy_decoder: Whether to load decoder weights on-demand to save memory
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier for feature visualization
    """
    
    def __init__(
        self,
        n_layers: int,
        d_transcoder: int,
        d_model: int,
        activation_function: str = "relu",
        lazy_decoder = True,
        lazy_encoder = False,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
        scan: str | list[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        clt_path: str | None = None,
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        
        self.n_layers = n_layers
        self.d_transcoder = d_transcoder
        self.d_model = d_model
        self.lazy_decoder = lazy_decoder
        self.lazy_encoder = lazy_encoder
        self.clt_path = clt_path
        
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.skip_connection = False
        self.scan = scan
        
        if activation_function == "jump_relu":
            self.activation_function = JumpReLU(
                torch.zeros(n_layers, 1, d_transcoder, device=device, dtype=dtype)
            )
        elif activation_function == "relu":
            self.activation_function = F.relu
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")
        
        if not lazy_encoder:
            self.W_enc = nn.Parameter(
                torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
            )
            
        self.b_dec = torch.nn.Parameter(torch.zeros(n_layers, d_model, device=device, dtype=dtype))
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)
        )
        
        if not lazy_decoder:
            self.W_dec = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(d_transcoder, n_layers - i ,d_model, device=device, dtype=dtype)
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.W_dec = None
            
# Unfinished. See: https://github.com/safety-research/circuit-tracer/blob/main/circuit_tracer/transcoder/cross_layer_transcoder.py