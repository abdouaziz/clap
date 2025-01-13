from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Any, Dict
from clap import DataCollatorWithPadding , CLAP, CLAPTrainer
from transformers import Wav2Vec2Processor, AutoFeatureExtractor, AutoTokenizer, HfArgumentParser


@dataclass
class ModelArguments:
    text_encoder_name: str = field(
        metadata={"help": "The name or path of the text encoder model"},
        default="google-bert/bert-base-cased"
    )
    audio_encoder_name: str = field(
        metadata={"help": "The name or path of the audio encoder model"},
        default="facebook/wav2vec2-base"
    )
    projection_dim: int = field(
        metadata={"help": "The dimension of the projection head"},
        default=768
    )
    hf_dataset_name: str = field(
        metadata={"help": "The name of the Hugging Face dataset"},
        default="abdouaziiz/alffa_clap"
    )
    batch_size: int = field(
        metadata={"help": "The batch size"},
        default=1
    )
    num_epochs: int = field(
        metadata={"help": "The number of epochs"},
        default=100
    )
    learning_rate: float = field(
        metadata={"help": "The learning rate"},
        default=1e-4
    )
    weight_decay: float = field(
        metadata={"help": "The weight decay"},
        default=0.01
    )
    max_grad_norm: float = field(
        metadata={"help": "The maximum gradient norm"},
        default=1.0
    )
    use_wandb: bool = field(
        metadata={"help": "Whether to use wandb for logging"},
        default=True
    )
    project_name: str = field(
        metadata={"help": "The name of the wandb project"},
        default="CLAP_training"
    )
    checkpoint_dir: str = field(
        metadata={"help": "The directory to save checkpoints"},
        default="./checkpoints"
    )

def main():
    parser = HfArgumentParser((ModelArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        # Process audio input
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        # Process transcription labels
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcription"]).input_ids
        return batch

    model = CLAP(
        text_encoder_name=args.text_encoder_name,
        audio_encoder_name=args.audio_encoder_name,
        projection_dim=args.projection_dim,
    )

    train_dataset = load_dataset(args.hf_dataset_name, split="train")
    val_dataset = load_dataset(args.hf_dataset_name, split="test")

    train_dataset = train_dataset.map(
        prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4
    )
    val_dataset = val_dataset.map(
        prepare_dataset, remove_columns=val_dataset.column_names, num_proc=4
    )

    data_collator = DataCollatorWithPadding(processor=processor, padding=True)

    train_loader = DataLoader(
        train_dataset, 
        collate_fn=data_collator, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        collate_fn=data_collator, 
        batch_size=args.batch_size, 
        num_workers=4
    )

    # Create trainer and train
    trainer = CLAPTrainer(model, train_loader, val_loader, args)
    trainer.train()

if __name__ == "__main__":
    main()