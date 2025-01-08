import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging
from typing import Dict
import os



def contrastive_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Compute NT-Xent loss for CLAP
    """
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size, device=similarity.device)
    
    # Symmetric loss
    loss_i = F.cross_entropy(similarity, labels)
    loss_t = F.cross_entropy(similarity.T, labels)
    
    return (loss_i + loss_t) / 2.0

class CLAPTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        if self.config['use_wandb']:
            wandb.init(
                project=self.config['project_name'],
                config=self.config
            )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, path)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Move batch to device
            audio = batch['audio'].to(self.device)
            text = batch['text']
            
            # Forward pass
            similarity, _, _ = self.model(text, audio)
            
            # Calculate loss
            loss = contrastive_loss(similarity)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if self.config['use_wandb']:
                wandb.log({'train_batch_loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            audio = batch['audio'].to(self.device)
            text = batch['text']
            
            # Forward pass
            similarity, _, _ = self.model(text, audio)
            
            # Calculate loss
            loss = contrastive_loss(similarity)
            total_loss += loss.item()
            
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            logging.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch()
            logging.info(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate()
            logging.info(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if self.config['use_wandb']:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
            logging.info(f"Best validation loss: {best_val_loss:.4f}")

