import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class CoLenNet(nn.Module):
    def __init__(
            self, 
            text_embed_dim, 
            motion_embed_dim, 
            hidden_dim,
            clip_dim=512,
            word_emb_dim=300,
            pos_emb_dim=300,
            dropout_p=0.1,
            use_ours=False
        ):
        super().__init__()
        # Projection heads for contrastive learning
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_embed_dim, hidden_dim)

        # Classifier for motion length prediction
        self.length_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )

        self.use_ours = use_ours
        if use_ours:
            self.cond_proj = nn.Linear(clip_dim, hidden_dim)
            self.pos_emb_proj = nn.Embedding(22 + 2, pos_emb_dim)
            # Multihead attention
            self.cross_attn = nn.MultiheadAttention(embed_dim=word_emb_dim, num_heads=4, dropout=dropout_p)
            self.word_emb_proj = nn.Linear(word_emb_dim, clip_dim)
            
            self.text_position_enc = PositionalEncoding(hidden_dim, dropout_p, max_len=22+3)
            self.clip_word_norm = nn.LayerNorm(hidden_dim)  

    def forward(self, sen_emb, motion_feat=None, word_emb=None, pos=None):
        if self.use_ours:
            # POS embedding
            pos_emb = self.pos_emb_proj(pos)  # (B, L, pos_emb_dim)

            # Q = word embedding, K = V = pos embedding
            Q = word_emb.transpose(0, 1)
            K = pos_emb.transpose(0, 1)
            V = pos_emb.transpose(0, 1)

            attn_output, _ = self.cross_attn(Q, K, V)      # (L, B, D)
            word_pos_attn = attn_output.transpose(0, 1)    # (B, L, D)
            
            word_attn_emb = self.word_emb_proj(word_pos_attn)      # (B, L, 300) → (B, L, 512)
            sen_emb = sen_emb.unsqueeze(1)                       # (B, 512) → (B, 1, 512)
            clip_word_pos = torch.cat([word_attn_emb, sen_emb], dim=1)  # (B, L+1, 512)
            
            clip_word_pos_emb = self.cond_proj(clip_word_pos)            # (B, L+1, hidden_dim)
            clip_word_pos_emb = clip_word_pos_emb.permute(1, 0, 2)       # (L+1, B, H)
            clip_word_pos_emb = self.text_position_enc(clip_word_pos_emb)
            clip_word_pos_emb = self.clip_word_norm(clip_word_pos_emb)

            # Step 2: Self-attention weight for pooling
            # weight_layer: shared across batches, so use one projection
            self_attn_weights = torch.mean(clip_word_pos_emb, dim=2)  # [L+1, B]
            self_attn_weights = F.softmax(self_attn_weights, dim=0)   # across time steps

            # Step 3: weight * feature → sum
            # (L+1, B, 1) * (L+1, B, H) → (L+1, B, H) → sum over L+1
            weighted = self_attn_weights.unsqueeze(-1) * clip_word_pos_emb  # (L+1, B, H)
            # Project to joint embedding space
            text_emb = weighted.sum(dim=0)  # (B, H)
        else:
            # Project to joint embedding space
            text_emb = self.text_proj(sen_emb)        # (bs, hidden)
        # Normalize for contrastive loss
        text_emb_norm = F.normalize(text_emb, dim=-1)
        
        if motion_feat is not None:
            motion_feat = motion_feat.mean(dim=1)
            motion_emb = self.motion_proj(motion_feat)  # (B, hidden)
            motion_emb_norm = F.normalize(motion_emb, dim=-1)
            motion_emb_norm = F.normalize(motion_emb, dim=-1)

        else:
            motion_emb_norm = None
        
        # Length classification logits
        length_pred = self.length_regressor(text_emb).squeeze(-1)  # [B]
        
        return text_emb_norm, motion_emb_norm, length_pred