from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from clap import prepare_dataset , DataCollatorWithPadding
from transformers import Wav2Vec2Processor, AutoFeatureExtractor, AutoTokenizer


feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(processor:Wav2Vec2Processor , batch: Dict[str, Any]) -> Dict[str, Any]:

    audio = batch["audio"]
    
    # Process audio input
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # Process transcription labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    
    return batch


def main():
 
    train = load_dataset("abdouaziiz/alffa_clap", split="train")
    #test = load_dataset("abdouaziiz/alffa_clap", split="test")
    print(train)
    
 
    train = train.map(
        prepare_dataset,
        remove_columns=train.column_names,
        num_proc=4
    )
    
 
    data_collator = DataCollatorWithPadding(processor=processor, padding=True)
    train_loader = DataLoader(
        train,
        collate_fn=data_collator,
        batch_size=2,
        num_workers=4
    )
    
 
    for batch in train_loader:
        print(batch)
        break


if __name__ == "__main__":
    main()