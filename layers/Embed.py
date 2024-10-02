import math
import torch
import torch.nn as nn


class LearnablePositionEncoding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
    ):
        super(LearnablePositionEncoding, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        return self.position_encoding


class AbsolutePositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super(AbsolutePositionEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        return self.pe[:, : x.size(1)]


class RotaryPositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(RotaryPositionEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        seq_len = x.shape[1]
        half_dim = self.d_model // 2
        position = torch.arange(
            seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, half_dim, 1, dtype=torch.float32, device=x.device)
            * -(math.log(10000.0) / half_dim)
        )
        angle_rads = position * div_term
        # Apply sin and cos to half of the dimensions
        sin_angle = torch.sin(angle_rads)
        cos_angle = torch.cos(angle_rads)
        x1 = x[..., 0::2] * cos_angle - x[..., 1::2] * sin_angle
        x2 = x[..., 0::2] * sin_angle + x[..., 1::2] * cos_angle
        x_rope = torch.cat([x1, x2], dim=-1)
        return x_rope


class Patch(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
    ):
        super(Patch, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        :param x: [batch_size * num_features, input_len, 1]
        :return: [batch_size * num_features, num_patches, d_model]
                num_patches = seq_len = (input_len - patch_len) // stride + 1
        """
        x = x.squeeze(-1)  # [batch_size * num_features, input_len]
        x = x.unfold(-1, self.patch_len, self.stride)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_len: int,
        d_model: int,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_embedding = nn.Linear(patch_len, d_model, bias=False)

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, patch_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        x = self.patch_embedding(x)
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.token_embedding(x.transpose(1, 2)).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
    ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position_encoding = AbsolutePositionEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        x = x + self.position_encoding(x)
        return self.dropout(x)


class DePatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_len: int,
        d_model: int,
    ):
        super(DePatchEmbedding, self).__init__()
        self.output_projection = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, patch_len]
        """
        x = self.output_projection(x)
        return x
