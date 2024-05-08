import torch
from torch import nn


class UNet3D(nn.Module):

    # This function creates a convolutional block with the given number of input and output features.
    def conv_block(self, in_feats, out_feats):
        return nn.Sequential(
            nn.InstanceNorm3d(in_feats),
            nn.Conv3d(in_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm3d(out_feats),
            nn.Conv3d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    # The init function defines the parameters that can be given to the U-Net.
    def __init__(self, out_channels, initial_features=32, in_channels=1, sigmoid_activation=True):
        super().__init__()

        # Define the number of channels ("features") throughout the U-Net. We increase the number of channels by a factor of 2
        # in each encoder level of the U-Net.
        # The first list defines the input channels of each level and the second the output channels.
        in_features = [in_channels, initial_features, initial_features * 2, initial_features * 4]
        out_features = [initial_features, initial_features * 2, initial_features * 4, initial_features * 8]

        # Build the encoder (= the lefthand side of the U-Net). Each encoder is a convolutional block.
        self.encoders = nn.ModuleList([
            self.conv_block(in_feats, out_feats) for in_feats, out_feats in zip(in_features, out_features)
        ])
        # Build the max-pooling layers that downsample the volume representation after each encoder block.
        self.poolers = [nn.MaxPool3d(2)] * len(in_features)
        # Build the base block (= the conv block that is applied at the bottom of the U-Net, after the encoders).
        self.base = self.conv_block(initial_features * 8, initial_features * 16)

        # Define the number of channels for the decoders of the U-Net. The decoder corresponds to the right-hand path
        # of the U-Net, where the volume representation is up-sampled again and combined with the representation from the
        # corresponding encoder level through the skip connections.
        in_features = [initial_features * 16, initial_features * 8, initial_features * 4, initial_features * 2]
        out_features = [initial_features * 8, initial_features * 4, initial_features * 2, initial_features]
        # Build the encoder blocks.
        self.decoders = nn.ModuleList([
            self.conv_block(in_feats, out_feats)
            for in_feats, out_feats in zip(in_features, out_features)
        ])
        # Build the upsampling layers that increase the spatial shape of the  volume representation
        # after each decoder layer. We use a transposed convolutional layer for upsampling.
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose3d(in_feats, out_feats, 2, stride=2)
            for in_feats, out_feats in zip(in_features, out_features)
        ])

        # Build the last convolutional layer that maps the predicted volume to the desired number of output channels.
        # We make use of a 1x1x1 convolution here.
        self.out_conv = nn.Conv3d(out_features[-1], out_channels, kernel_size=1)

        # Add the sigmoid activation if specified.
        # (This can be deactivated for loss functions that expect a different output range than [0, 1])
        self.sigmoid_activation = sigmoid_activation
        if sigmoid_activation is True:
            self.last_activation = nn.Sigmoid()

    # The "forward" function defines how the data flows through the network.
    def forward(self, x):
        # We first pass the input data through the encoder (left path) of the U-Net. After each
        # encoder block, we save the representation in the 'from_encoder' list, so that we can later
        # pass it to the corresponding decoder (these are the skip connections!) and then downsample the representation.
        from_encoder = []
        for encoder, pooler in zip(self.encoders, self.poolers):
            x = encoder(x)
            from_encoder.append(x)
            x = pooler(x)

        # Apply the base layer.
        x = self.base(x)

        # Then go through the decoder (right path of the U-Net).
        # Before the data goes into encoder we upsample it and concatenate the corresponding encoder representation,
        # which we saved in the 'from_encoder' list before.
        from_encoder = from_encoder[::-1]  # reverse the list so that it matches going from down to up.
        for decoder, upsampler, from_enc in zip(self.decoders, self.upsamplers, from_encoder):
            x = decoder(torch.cat([
                from_enc, upsampler(x)
            ], dim=1))

        # Apply the last convolution to map to the correct number of output channels.
        x = self.out_conv(x)
        # And apply the sigmoid activation if specified.
        if self.sigmoid_activation:
            x = self.last_activation(x)

        return x
