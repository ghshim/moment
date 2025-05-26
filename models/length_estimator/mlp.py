import torch
import torch.nn as nn

class CLIPAdapter(nn.Module):
    def __init__(self, input_dim=512, adapter_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)  # Residual connection
        

class CLIPAdapterLengthRegressor(nn.Module):
    def __init__(self, clip_dim=512, adapter_dim=256, hidden_dim=512):
        super().__init__()
        self.adapter = CLIPAdapter(clip_dim, adapter_dim)
        self.regressor = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, clip_emb):
        adapted = self.adapter(clip_emb)
        return self.regressor(adapted).squeeze(-1)  # [B]
