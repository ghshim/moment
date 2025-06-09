import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

from utils.utils import *

class CoLenTrainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model.to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.device = opt.device

        self.regression_criterion = nn.MSELoss()
        self.best_val_loss = float('inf')

        self.r_w = 0.4
        self.c_w = 0.6

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, train_loader, val_loader):
        wandb_init("MOMENT-LenEst", config=self.opt, id=self.opt.name)

        for epoch in tqdm(range(1, self.opt.num_epochs + 1)):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Validation Loss": val_loss})
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved at epoch {epoch} with val loss {val_loss:.4f}")

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for batch in loader:
            # caption, motion_feat, length_label, text_feat = batch
            word_embeddings, pos_one_hots, pos_indices, sen_embedding, caption, sent_len, motion, m_lens, tokens = batch
            
            text_feat = sen_embedding.to(self.device).float()
            pos_indices = pos_indices.to(self.device).long()
            word_embeddings = word_embeddings.to(self.device).float()
            motion_feat = motion.to(self.device).float()
            text_feat = sen_embedding.to(self.device).float()
            length_label = m_lens.to(self.device).float()

            self.optimizer.zero_grad()

            text_emb, motion_emb, length_pred = self.model(text_feat, motion_feat, word_emb=word_embeddings, pos=pos_indices)

            # Regression loss (length estimation)
            reg_loss = self.regression_criterion(length_pred, length_label)  # float target

            # Contrastive loss (InfoNCE)
            cont_loss = self.contrastive_loss(text_emb, motion_emb)

            loss = self.c_w * cont_loss + self.r_w * reg_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            wandb.log({"Regression Loss": reg_loss, "Contrastive Loss": cont_loss})

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                word_embeddings, pos_one_hots, pos_indices, sen_embedding, caption, sent_len, motion, m_lens, tokens = batch
            
                text_feat = sen_embedding.to(self.device).float()
                pos_indices = pos_indices.to(self.device).long()
                word_embeddings = word_embeddings.to(self.device).float()
                motion_feat = motion.to(self.device).float()
                text_feat = sen_embedding.to(self.device).float()
                length_label = m_lens.to(self.device).float()

                text_emb, motion_emb, length_pred = self.model(text_feat, motion_feat, word_emb=word_embeddings, pos=pos_indices)

                reg_loss = self.regression_criterion(length_pred, length_label)  # float target
                cont_loss = self.contrastive_loss(text_emb, motion_emb)

                loss = self.c_w * cont_loss + self.r_w * reg_loss
                total_loss += loss.item()

        return total_loss / len(loader)

    def contrastive_loss(self, text_emb, motion_emb, temperature=0.07):
        # text_emb, motion_emb: [B, D]
        batch_size = text_emb.size(0)
        sim_matrix = torch.matmul(text_emb, motion_emb.T) / temperature
        labels = torch.arange(batch_size).to(sim_matrix.device)

        loss_i2m = torch.nn.functional.cross_entropy(sim_matrix, labels)
        loss_m2i = torch.nn.functional.cross_entropy(sim_matrix.T, labels)

        return (loss_i2m + loss_m2i) / 2
