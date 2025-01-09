# CLAP: Contrastive Language-Audio Pretraining  

This repository accompanies the paper **"CLAP: Contrastive Language-Audio Pretraining for Universal Audio Understanding"**, which presents a novel framework for learning joint embeddings of language and audio through contrastive learning.  

## Overview  

CLAP introduces a powerful methodology for training models to understand and represent audio in a way that is aligned with natural language descriptions. By leveraging large-scale datasets and a contrastive learning objective, CLAP bridges the gap between audio and textual representations, enabling robust performance across a variety of downstream tasks.  

### Key Contributions  
- **Unified Audio-Language Embedding Space**: CLAP maps audio and language into a shared latent space, enabling natural language queries for audio retrieval and classification.  
- **Scalability**: The model is trained on a diverse and large-scale dataset, ensuring generalization across multiple domains.  
- **Multi-Task Applicability**: CLAP demonstrates state-of-the-art performance in tasks like zero-shot audio classification, captioning, and sound event detection.  


## Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/abdouaziz/clap.git
   cd clap