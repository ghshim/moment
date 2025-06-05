import torch
import torch.nn as nn
import torch.nn.functional as F

class CoLenNet(nn.Module):
    def __init__(self, text_embed_dim, motion_embed_dim, hidden_dim, num_length_classes):
        super().__init__()
        # Projection heads for contrastive learning
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_embed_dim, hidden_dim)

        # Classifier for motion length prediction
        self.length_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_length_classes)
        )

    def forward(self, text_feat, motion_feat):
        # Project to joint embedding space
        text_emb = self.text_proj(text_feat)        # (bs, hidden)
        motion_emb = self.motion_proj(motion_feat)  # (bs, hidden)

        # Normalize for contrastive loss
        text_emb_norm = F.normalize(text_emb, dim=-1)
        motion_emb_norm = F.normalize(motion_emb, dim=-1)

        # Length classification logits
        length_logits = self.length_classifier(text_emb)    
        
        return text_emb_norm, motion_emb_norm, length_logits