# GeoMap Model Documentation

## Overview
The `GeoMap` model is a modified Convolutional Conditional Neural Processes designed to predict oil reserves from contextual geological data. It leverages a combination of convolutional neural networks (CNNs), transformers, and deconvolutional networks to predict spatial probability maps.

## Model Architecture
The `GeoMap` model comprises several key components each responsible for a specific aspect of the learning process:

### Encoder
- **Type**: UNet-like architecture.
- **Function**: Extracts and encodes spatial features from the input well logs.
- **Details**: Utilizes convolutional layers to progressively downsample the input, capturing hierarchical features.

### Positional Embedding
- **Type**: 2D Positional Embedding.
- **Function**: Provides the model with information about the relative or absolute position of the samples in the input data.
- **Details**: Generates embeddings that are added to the encoder outputs to maintain spatial awareness in subsequent layers.

### Transformer
- **Type**: Stacked Transformers.
- **Function**: Processes the encoded and embedded input to integrate contextual information across different wells.
- **Details**: Applies self-attention mechanisms to model dependencies without regard to their distance in the input data.

### Fully Connected Series
- **Type**: Series of Linear Layers.
- **Function**: Transforms the output of the transformer into a feature space suitable for generating spatial outputs.
- **Details**: Maps the high-level features derived from the transformer to a space where spatial dimensions can be reconstructed.

### Decoder
- **Type**: Deconvolutional Network.
- **Function**: Up-samples and reconstructs the final output to the desired spatial resolution, predicting well logs.
- **Details**: Uses transpose convolutional layers to progressively upscale the feature maps to the target resolution.

### Activation
- **Type**: Sigmoid.
- **Function**: Normalizes the final output to a range suitable for interpretation as log probabilities or similar metrics.

## Input and Output
- **Input**: 
  - `context`: A tensor of shape `(batch_size, num_wells, in_channels, depth)` containing the well logs and contextual geological data.
  - `Cx`: Coordinates for positional embeddings, shape `(batch_size, num_wells, 2)`.
- **Output**:
  - A tensor of predicted well logs with shape `(batch_size, output_height, output_width)`.
 
