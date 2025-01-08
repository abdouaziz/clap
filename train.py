from clap import CLAP, CLAPTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader


def main():

    config = {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "use_wandb": True,
        "project_name": "CLAP_training",
        "checkpoint_dir": "./checkpoints",
        "text_encoder_name": "bert-base-uncased",
        "audio_encoder_name": "facebook/wav2vec2-base",
        "projection_dim": 512,
    }

    model = CLAP(
        text_encoder_name=config["text_encoder_name"],
        audio_encoder_name=config["audio_encoder_name"],
        projection_dim=config["projection_dim"],
    )

    train_dataset = load_dataset("", split="train")
    val_dataset = load_dataset("", split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Create trainer and train
    trainer = CLAPTrainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
