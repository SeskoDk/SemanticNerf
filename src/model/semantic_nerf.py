import torch
import torch.nn as nn
import tinycudann as tcnn


def trunc_exp(x):
    # Clamp the range to prevent overflow in exp(x)
    return torch.exp(torch.clamp(x, max=20))


class SEM_NGP(nn.Module):

    def __init__(
        self,
        nerf_type="large",
        density_ch=1,
        semantic_ch=2,  # semantic classes
        color_ch=3,
        geo_feat_dim=15,
        bound=1,
    ):
        super().__init__()

        self.bound = bound
        mlp_type = tcnn_mlp_type()
        self.semantic_channels = semantic_ch

        # Hash-table size selector
        if nerf_type == "small":
            log2_size = 15
        elif nerf_type == "medium":
            log2_size = 17
        elif nerf_type == "large":
            log2_size = 19

        # -----------------------------------------------------------
        # POSITION ENCODING (HashGrid)
        # -----------------------------------------------------------
        self.pos_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_size,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
        )

        # -----------------------------------------------------------
        # DIRECTION ENCODING (Spherical Harmonics)
        # -----------------------------------------------------------
        self.dir_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,  # output = 16 dims
            },
        )

        # GEOMETRY / DENSITY NETWORK (sigma + geo_feat)
        self.dmlp = tcnn.Network(
            n_input_dims=self.pos_encoding.n_output_dims,
            n_output_dims=density_ch + geo_feat_dim,
            network_config={
                "otype": mlp_type,
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        # COLOR NETWORK (RGB, view-dependent)
        self.cmlp = tcnn.Network(
            n_input_dims=self.dir_encoding.n_output_dims + geo_feat_dim,
            n_output_dims=color_ch,  # RGB
            network_config={
                "otype": mlp_type,
                "activation": "ReLU",
                "output_activation": "Sigmoid",  # normalized RGB
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        # SEMANTIC NETWORK (view-INdependent)
        # Multiclass output = C classes
        self.sem_mlp = tcnn.Network(
            n_input_dims=geo_feat_dim,  # NO direction information
            n_output_dims=self.semantic_channels,  # semantic classes
            network_config={
                "otype": mlp_type,
                "activation": "ReLU",
                "output_activation": "None",  # logits for CE loss
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def forward(self, pos, viewdirs):
        # pos: [N, 3], in [-bound, bound]
        # viewdirs: [N, 3], nomalized in [-1, 1]

        # sigma
        # pos = (pos + self.bound) / (2 * self.bound)  # to [0, 1]
        # Encode 3D coordinates
        pos_encoded = self.pos_encoding(pos)
        # Density + geometry features
        h = self.dmlp(pos_encoded)

        # sigma = h[:, :1]  # (B,1)
        sigma = trunc_exp(h[:, :1])  # (B,1)
        geo_feat = h[..., 1:]  # (B, geo_feat_dim)

        # color
        viewdirs = (
            viewdirs + 1
        ) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        # Encode viewing direction
        dir_encoded = self.dir_encoding(viewdirs)

        # RGB uses view direction + geometry features
        color_input = torch.cat([geo_feat, dir_encoded], dim=-1)
        rgb = self.cmlp(color_input)

        # Semantic logits (MULTICLASS)
        # shape: (B, semantic_ch)
        sem_logits = self.sem_mlp(geo_feat)

        # RETURN:
        #  rgb         → (B,3)
        #  sigma       → (B,1)
        #  sem_logits  → (B,C)
        return rgb, sigma, sem_logits


def tcnn_mlp_type() -> str:
    """
    Select the best tiny-cuda-nn MLP type for the current GPU.
    """
    if not torch.cuda.is_available():
        return "CutlassMLP"

    cc_major, cc_minor = torch.cuda.get_device_capability()
    cc = cc_major * 10 + cc_minor

    return "FullyFusedMLP" if cc >= 75 else "CutlassMLP"


if __name__ == "__main__":
    # Simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEM_NGP().to(device)

    N = 10
    positions = torch.rand(N, 3, device=device) * 2 - 1  # in [-1, 1]
    viewdirs = torch.rand(N, 3, device=device) * 2 - 1  # in [-1, 1]
    rgb, sigma, sem_logits = model(positions, viewdirs)
    print("RGB:", rgb.shape)  # (N,3)
    print("Sigma:", sigma.shape)  # (N,1)
    print("Sem_logits:", sem_logits.shape)  # (N,C)
