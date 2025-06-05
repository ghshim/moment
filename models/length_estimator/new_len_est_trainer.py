import torch
import os
from tqdm import tqdm
import wandb

class CoLengthTrainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model.to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.device = opt.device

        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, train_loader, val_loader):
        for epoch in tqdm(range(1, self.opt.num_epochs + 1)):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

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
            caption, motion_feat, length_label, text_feat = batch
            
            text_feat = text_feat.to(self.device).float()
            motion_feat = motion_feat.to(self.device).float()
            length_label = length_label.to(self.device).long()

            self.optimizer.zero_grad()

            text_emb, motion_emb, logits = self.model(text_feat, motion_feat)

            # Classification loss
            cls_loss = self.cls_criterion(logits, length_label)

            # Contrastive loss (InfoNCE)
            cont_loss = self.contrastive_loss(text_emb, motion_emb)

            loss = cls_loss + cont_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                caption, motion_feat, length_label, text_feat = batch
                
                text_feat = text_feat.to(self.device).float()
                motion_feat = motion_feat.to(self.device).float()
                length_label = length_label.to(self.device).long()

                text_emb, motion_emb, logits = self.model(text_feat, motion_feat)

                cls_loss = self.cls_criterion(logits, length_label)
                cont_loss = self.contrastive_loss(text_emb, motion_emb)

                loss = cls_loss + cont_loss
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
