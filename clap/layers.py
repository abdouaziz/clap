import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertForPreTraining
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)


class TextEncoder(nn.Module):
    def __init__(self, path_or_model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_model_name)
        self.model = BertForPreTraining.from_pretrained(path_or_model_name)

    def forward(self, text: str):
        inputs = self.tokenizer(text=text, return_tensors="pt")
        outputs = self.model(**inputs)
        # Get the last hidden states
        hidden_states = (
            outputs.hidden_states[-1]
            if isinstance(outputs.hidden_states, tuple)
            else outputs.hidden_states
        )
        return hidden_states


class AudioEncoder(nn.Module):
    def __init__(self, path_or_model_name):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            path_or_model_name
        )
        self.model = Wav2Vec2ForPreTraining.from_pretrained(path_or_model_name)

    @torch.no_grad()
    def forward(self, array):
        # Process input audio
        input_values = self.feature_extractor(array, return_tensors="pt").input_values
        batch_size, raw_sequence_length = input_values.shape

        # Get sequence length after feature extraction
        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        ).item()

        # Compute mask indices
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        )

        # Sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        # Convert to tensors
        mask_time_indices = torch.tensor(
            data=mask_time_indices, device=input_values.device, dtype=torch.long
        )
        sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        )

        # Get model outputs
        outputs = self.model(
            input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )

        # Get the last hidden states
        hidden_states = (
            outputs.hidden_states[-1]
            if isinstance(outputs.hidden_states, tuple)
            else outputs.hidden_states
        )
        return hidden_states


class CLAP(nn.Module):
    def __init__(self, text_encoder_name, audio_encoder_name, projection_dim=512):
        super().__init__()
        self.text_encoder = TextEncoder(text_encoder_name)
        self.audio_encoder = AudioEncoder(audio_encoder_name)

        # Add projection heads
        self.text_projection = nn.Linear(
            self.text_encoder.model.config.hidden_size, projection_dim
        )
        self.audio_projection = nn.Linear(
            self.audio_encoder.model.config.hidden_size, projection_dim
        )

        # Temperature parameter for loss scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text, audio):
        # Get embeddings
        text_features = self.text_encoder(text)
        audio_features = self.audio_encoder(audio)

        # Apply projection
        text_embeddings = self.text_projection(text_features)
        audio_embeddings = self.audio_projection(audio_features)

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        # Calculate similarity
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * torch.matmul(
            text_embeddings, audio_embeddings.transpose(0, 1)
        )

        return similarity, text_embeddings, audio_embeddings
