import torch
import os
import numpy as np
from tqdm import tqdm
import wandb

class LengthEstimatorTrainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model.to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.criterion = torch.nn.MSELoss()
        self.device = opt.device

        self.best_val_loss = float('inf')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, train_loader, val_loader):
        for epoch in tqdm(range(1, self.opt.num_epochs + 1)):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
            caption, motion, m_length, embedding = batch
            
            embedding = embedding.to(self.device).float()
            m_length = m_length.to(self.device).float()

            self.optimizer.zero_grad()
            length_pred = self.model(embedding)  # [B]
            loss = self.criterion(length_pred, m_length)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        return avg_loss

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                caption, motion, m_length, embedding = batch
            
                embedding = embedding.to(self.device).float()
                m_length = m_length.to(self.device).float()

                length_pred = self.model(embedding)
                loss = self.criterion(length_pred, m_length)
                total_loss += loss.item()
        print(length_pred[0], length_pred[10], length_pred[20])
        print(m_length[0], m_length[10], m_length[20])
                

        avg_loss = total_loss / len(loader)
        return avg_loss
