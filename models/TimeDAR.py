import torch
import torch.nn as nn
from argparse import Namespace
from layers.TimeDAR_backbone import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    ForecastingHead,
)
from layers.Embed import (
    Patch,
    PatchEmbedding,
    TokenEmbedding,
    PositionalEncoding
)


class TimeDAR(nn.Module):
    def __init__(self, args: Namespace):
        super(TimeDAR, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.feedforward_dim = args.feedforward_dim
        self.dropout = args.dropout
        self.num_layers_casual = args.num_layers_casual
        self.num_layers_denoising = args.num_layers_denoising
        self.device = args.device
        self.task_name = args.task_name

        # Channel Independence
        self.channel_independence = ChannelIndependence(
            input_len=self.input_len,
        )

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        # Embedding
        if args.embedding == "patch":
            self.patch_embedding = PatchEmbedding(
                patch_len=self.patch_len,
                d_model=self.d_model,
            )
        elif args.embedding == "token":
            self.patch_embedding = TokenEmbedding(
                in_channels=self.patch_len,
                d_model=self.d_model,
            )
        else:
            raise ValueError("embedding should be one of ['patch', 'token']")

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        # Add SOS token and drop last
        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Casual Transformer
        self.casual_transformer = CausalTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            num_layers=self.num_layers_casual,
        )

        # Diffusion
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )

        # Decoder
        # Denoising Patch Decoder
        self.denoising_patch_decoder = DenoisingPatchDecoder(
            d_model=self.d_model,
            num_layers=self.num_layers_denoising,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )

        # Decoder
        self.decoder = ForecastingHead(
            seq_len=self.seq_len,
            d_model=self.d_model,
            pred_len=args.input_len,
            dropout=args.head_dropout,
        )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        # Instance Normalization
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]

        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # For Casual Transformer
        x_embedding = self.patch_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(
            x_embedding_bias
        )
        x_out = self.casual_transformer(
            x_embedding_bias,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(x_patch)  # [batch_size * num_features, seq_len, patch_len] 
        noise_x_embedding = self.patch_embedding(
            noise_x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(
            noise_x_embedding
        )

        # For Denoising Patch Decoder
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]

        # For Decoder
        predict_x = predict_x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        predict_x = self.decoder(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]

        return predict_x

    def forecast(self, x):
        # [batch_size, input_len, num_features]
        # Channel Independence And Patch
        batch_size, _, num_features = x.size()
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        x = self.patch_embedding(x)  # [batch_size * num_features, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]
        x = self.casual_transformer(
            x,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]
        x = x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        return x

    def forward(self, x, task):
        if task == "pretrain":
            return self.pretrain(x)  # [batch_size * num_features, seq_len, patch_len]
        elif task == "forecast":
            return self.forecast(x)  # [batch_size, num_features, seq_len, d_model]


class TimeDARForecasting(nn.Module):
    def __init__(self, args: Namespace, TimeDAR_encoder: TimeDAR):
        super(TimeDARForecasting, self).__init__()

        self.pred_len = args.pred_len
        self.TimeDAR_encoder = TimeDAR_encoder

        self.seq_len = int((args.input_len - args.patch_len) / args.stride) + 1
        self.forecasting_head = ForecastingHead(
            seq_len=self.seq_len,
            d_model=args.d_model,
            pred_len=args.pred_len,
            dropout=args.finetune_head_dropout,
        ).to(args.device)

    def forward(self, x, finetune_mode: str = "fine_all"):
        if finetune_mode == "fine_all":
            # Instance Normalization
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs

            x = self.TimeDAR_encoder(
                x, task="forecast"
            )  # (batch_size, num_features, seq_len, d_model)
            x = self.forecasting_head(x)  # (batch_size, pred_len, num_features)

            # Instance Denormalization
            x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        elif finetune_mode == "fine_last":
            with torch.no_grad():
                # Instance Normalization
                means = torch.mean(x, dim=1, keepdim=True).detach()
                x = x - means
                stdevs = torch.sqrt(
                    torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x = x / stdevs

                x = self.TimeDAR_encoder(x, task="forecast")
            x = self.forecasting_head(x)
            with torch.no_grad():
                x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
                x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        else:
            raise ValueError(
                "fine_tuning_mode should be one of ['fine_all', 'fine_last']"
            )
        return x

    def freeze_encoder(self):
        for param in self.TimeDAR_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.TimeDAR_encoder.parameters():
            param.requires_grad = True
