from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from clap import  DataCollatorWithPadding , AudioEncoder , TextEncoder
from transformers import Wav2Vec2Processor, AutoFeatureExtractor, AutoTokenizer


feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:

    audio = batch["audio"]
    
    print(f"The sampling rate of the audio is: {audio['sampling_rate']}")
    
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
    audio_encoder = AudioEncoder("facebook/wav2vec2-base")
    text_encoder = TextEncoder("google-bert/bert-base-cased")
 
    train = load_dataset("abdouaziiz/alffa_clap", split="train")
    #test = load_dataset("abdouaziiz/alffa_clap", split="test")

 
    train = train.map(
        prepare_dataset,
        remove_columns=train.column_names,
        num_proc=4
    )
    
 
    data_collator = DataCollatorWithPadding(processor=processor, padding=True)
    train_loader = DataLoader(
        train,
        collate_fn=data_collator,
        batch_size=1,
        num_workers=4
    )
    
 
    for batch in train_loader:
        #input_values=torch.tensor(batch["input_values"])
        labels=torch.tensor(batch["labels"])
        #outputs = audio_encoder(input_values)
        outputs = text_encoder({"input_ids": labels})
        print(outputs.shape)
        
        
        break


if __name__ == "__main__":
    main()