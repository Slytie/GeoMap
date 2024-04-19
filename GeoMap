import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, V), p_attn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = Q.size(0)

        # Linearly project the inputs
        Q, K, V = [l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (Q, K, V))]

        # Apply scaled dot-product attention
        x, self.attn = ScaledDotProductAttention(self.dropout.p)(Q, K, V, mask=mask)

        # Concatenate and project the output
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        return self.linears[-1](x)



class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        mha_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(mha_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x



class StackedTransformers(nn.Module):
    """Defines a stack of transformer blocks."""

    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super(StackedTransformers, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionalEmbedding2D(torch.nn.Module):
    """
    A PyTorch module to generate and provide 2D positional embeddings.

    This module creates positional embeddings for a specified 2D grid size and embedding dimension.
    It is designed to handle inputs of shape `(batch_size, sample, x_position, y_position)` and
    returns embeddings of shape `(batch, sample, dimension)`.

    Attributes:
        pe (torch.Tensor): The positional embeddings tensor.

    Methods:
        forward(positions): Returns the positional embeddings for given positions.
    """

    def __init__(self, grid_size, d_model):
        """
        Initializes the 2D positional embeddings.

        Args:
            grid_size (tuple): A tuple (x_max, y_max) defining the size of the grid.
            d_model (int): The embedding dimension. Must be an even number.

        Raises:
            AssertionError: If `d_model` is not an even number.
        """
        super().__init__()
        assert d_model % 2 == 0, "Embedding dimension must be even"
        x_max, y_max = grid_size
        self.pe = torch.zeros(x_max, y_max, d_model)

        # Prepare division term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        for x in range(x_max):
            for y in range(y_max):
                # Even indices: x
                self.pe[x, y, 0:d_model//2:2] = torch.sin(x * div_term[0:d_model//4])
                self.pe[x, y, 1:d_model//2:2] = torch.cos(x * div_term[0:d_model//4])

                # Odd indices: y
                self.pe[x, y, d_model//2::2] = torch.sin(y * div_term[d_model//4:])
                self.pe[x, y, d_model//2+1::2] = torch.cos(y * div_term[d_model//4:])

    def forward(self, positions):
        """
        Retrieves the positional embeddings for given positions.

        Args:
            positions (torch.Tensor): A tensor of shape `(batch_size, sample, 2)` where the last dimension
                                      contains x and y coordinates.

        Returns:
            torch.Tensor: The positional embeddings of shape `(batch, sample, dimension)`.
        """
        batch_size, sample_size, _ = positions.size()
        positions = positions.long()  # Ensure integer indices

        # Retrieve the positional embeddings for each sample in the batch
        embeddings = torch.zeros(batch_size, sample_size, self.pe.size(-1))
        for i in range(batch_size):
            for j in range(sample_size):
                x, y = positions[i, j]
                embeddings[i, j] = self.pe[x, y]

        return embeddings



class ResidualConv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0.25):
        super(ResidualConv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

        # Adjusting match_dimensions to ensure dimension matching
        self.match_dimensions = None
        if stride != 1 or in_channels != out_channels:
            self.match_dimensions = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout

        # Residual connection
        if self.match_dimensions is not None:
            identity = self.match_dimensions(identity)
        out = out + identity
        out = self.relu(out)

        return out


# UNet-like Residual 1D Model
class UNetResidual1D(nn.Module):
    def __init__(self):
        super(UNetResidual1D, self).__init__()
        # Initial depth-wise convolution
        self.initial_conv = ResidualConv1DBlock(8, 128, 3, 1, 1, dropout_rate=0.1)

        # Down-sampling layers
        self.downsample1 = ResidualConv1DBlock(128, 64, 3, 2, 1, dropout_rate=0.1)
        self.downsample2 = ResidualConv1DBlock(64, 64, 3, 2, 1, dropout_rate=0.1)
        self.downsample3 = ResidualConv1DBlock(64, 32, 3, 2, 1, dropout_rate=0.1)

        # Adjusting layers to further reduce the length
        self.adjust_length1 = ResidualConv1DBlock(32, 16, 3, 2, 1, dropout_rate=0.1)
        self.adjust_length2 = ResidualConv1DBlock(16, 8, 3, 2, 1, dropout_rate=0.1)

        self.final_fc = nn.Linear(8 * 13, 32)


    def forward(self, x):
        x = self.initial_conv(x)
        #print(x.shape) #torch.Size([32, 128, 400])
        x = self.downsample1(x)
        #print(x.shape) # torch.Size([32, 64, 200])
        x = self.downsample2(x)
        #print(x.shape) # torch.Size([32, 64, 100])
        x = self.downsample3(x)
        #print(x.shape) # torch.Size([32, 32, 50])
        x = self.adjust_length1(x)
        #print(x.shape) # torch.Size([32, 16, 25])
        x = self.adjust_length2(x)
        # Flatten the output from the last convolutional block
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, feature_size)
        #print(x.shape) # torch.Size([32, 8, 13])
        # Pass through the final fully connected layer to get to the desired shape
        x = self.final_fc(x)
        #print(x.shape) # torch.Size([32, 8, 13])
        return x

class SimpleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0.1, use_bn=True):
        super(SimpleConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.initial_conv = SimpleConv1D(8, 128, 2, 1, 1)
        self.downsample1 = SimpleConv1D(128, 64, 3, 2, 1)
        self.downsample2 = SimpleConv1D(64, 64, 3, 2, 1)
        self.downsample3 = SimpleConv1D(64, 32, 3, 2, 1)
        self.adjust_length1 = SimpleConv1D(32, 16, 3, 2, 1)
        self.adjust_length2 = SimpleConv1D(16, 8, 3, 2, 1)
        self.final_fc = nn.Linear(8 * 13, 32)  # Adjusted to match the output size

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.adjust_length1(x)
        x = self.adjust_length2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.final_fc(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batchnorm, dropout_rate=0.20):
        super(DeconvBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


class DeconvNet(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.25):
        super(DeconvNet, self).__init__()
        self.relu = nn.ReLU()
        # Initial convolutional layer to process input at 32x32
        self.initial_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        # Upscale 32x32 to 64x64
        self.deconv1 = DeconvBlock(1, 32, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm,
                                   dropout_rate=0.10)

        # Additional convolutional layer to process at 64x64
        self.process_64 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Upscale 64x64 to 128x128
        self.deconv2 = DeconvBlock(32, 1, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm,
                                   dropout_rate=0.10)

        # Downscale to 96x96 using adaptive pooling
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))

        self.deconv3 = DeconvBlock(1, 1, kernel_size=3, stride=1, padding=1, use_batchnorm=False, dropout_rate=0)

        # Additional layers to add complexity at 96x96
        self.complex_conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.complex_conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        #print(x.shape) # torch.Size([32, 1, 16, 16])
        #x = self.relu(x)
        x = self.deconv1(x) # torch.Size([32, 32, 32, 32])
        #print(x.shape)
        x = self.process_64(x) # torch.Size([32, 32, 32, 32])
        #print(x.shape)
        #x = self.relu(x)
        x = self.deconv2(x) # torch.Size([32, 16, 33, 33])
        #print(x.shape)
        #x = self.adaptive_pool(x) #torch.Size([32, 16, 64, 64])
        x = self.deconv3(x) # torch.Size([32, 8, 64, 64])
        #print(x.shape)
        x = self.complex_conv1(x) # torch.Size([32, 16, 64, 64])
        #print(x.shape)
        #x = self.relu(x)
        x = self.complex_conv2(x) # torch.Size([32, 1, 64, 64])
        #print(x.shape)
        #x = self.relu(x)
        return x


class FullyConnectedSeries(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=0, leaky_relu_slope=0.01):
        """
        Initializes a series of fully connected layers.

        Parameters:
        - layer_sizes (list): A list containing the sizes of each layer.
        - dropout_prob (float): Probability of dropout (default: 0.5).
        - leaky_relu_slope (float): Negative slope coefficient for LeakyReLU (default: 0.01).
        """
        super(FullyConnectedSeries, self).__init__()
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))

            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)


class GeoMap(nn.Module):
    """Implements a modified Conditional Neural Process (GeoMap) model for predicting the locations of oil reserves.

    The model incorporates an encoder, transformer, and decoder to process and predict
    well log data based on the input well logs and contextual information. It uses
    a UNet-style encoder for initial feature extraction, positional embeddings for capturing
    spatial relationships, a transformer for integrating context across different wells,
    and a decoder comprising a fully connected layer followed by a deconvolutional network
    for final prediction output.

    Attributes:
        encoder (nn.Module): Encoder module using a UNet architecture to encode input tensors.
        stacked_transformer (nn.Module): Transformer module to process encoded inputs.
        positional_embedding (nn.Module): Module to generate and apply 2D positional embeddings.
        sigmoid (nn.Module): Sigmoid activation layer.
        deconv (nn.Module): Deconvolution network to transform the data back to the spatial domain.
        fc (nn.Module): Fully connected series to process the transformer outputs before decoding.

    Architecture Design Choices:
        - The encoder is a simplified UNet that uses convolutional layers for feature extraction.
        - Positional embeddings add context related to the positions within the input grid.
        - The stacked transformer applies attention mechanisms to integrate features across different contexts.
        - The final output shape is adjusted via deconvolutional layers after processing through a fully connected series.

    Args:
        N/A: Constructor takes no arguments as the model's architecture specifics are hardcoded.

    Input Shape:
        context (Tensor): The input tensor of shape (batch_size, num_wells, in_channels, depth).
        Cx (Tensor): Coordinates for positional embeddings, shape (batch_size, num_wells, 2).

    Returns:
        Tensor: Output tensor of predicted well logs with shape (batch_size, output_height, output_width).
    """

    def __init__(self):
        super(GeoMap, self).__init__()

        self.encoder = SimpleUNet()
        self.stacked_transformer = StackedTransformers(d_model=32, num_heads=8, num_layers=8, dropout=0.10)
        self.positional_embedding = PositionalEmbedding2D(grid_size=(64, 64), d_model=32)
        self.sigmoid = nn.Sigmoid()
        self.deconv = DeconvNet()
        self.fc = FullyConnectedSeries([384, 256])
        #self.fc = FullyConnectedSeries([192, 100, 100, 192])


    def forward(self, context, Cx):
        # Creating a new tensor with the desired final shape
        cropped_context = torch.zeros(context.size(0), context.size(1), context.size(2), 400)

        # Cropping operation with direct replacement in the new tensor
        for i in range(context.size(0)):
            for j in range(context.size(1)):
                start_index = random.randint(50, 100)  # Ensuring start_index is within bounds
                end_index = start_index + 400
                cropped_context[i, j, :, :] = context[i, j, :, start_index:end_index]

        context = cropped_context

        random_indices = torch.randperm(context.size(1))
        embeddings = self.positional_embedding(Cx)
        embeddings[:, :, 0:2] += Cx
        encoded_wells = [self.encoder(context[:, i]).view(context.shape[0], -1) + embeddings[:, i] for i in random_indices]
        #encoded_wells = [self.encoder(context[:, i]).view(context.shape[0], -1) for i in random_indices]

        encoded_context = torch.stack(encoded_wells, dim=1)
        output = self.stacked_transformer(encoded_context)
        #print(output.shape)
        encoded_context = encoded_context.view(encoded_context.shape[0], -1)
        #print(encoded_context.shape)
        output = self.fc(encoded_context)
        #print(encoded_context.shape)
        output = output.view(output.shape[0], 16, 16)
        output = output.unsqueeze(1)
        output = self.deconv(output)
        output = output.squeeze(1)
        output = output.view(output.shape[0], 64, 64)
        output = self.sigmoid(output)
        return output


