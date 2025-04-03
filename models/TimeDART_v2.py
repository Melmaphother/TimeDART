import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding
import os


class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        dropout: float,
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x


class Model(nn.Module):
    """
    TimeDART v2 with Qwen2.5-0.5B encoder
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.args = args

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.channel_independence = ChannelIndependence()

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Initialize encoder (Qwen2.5-0.5B)
        self.device = self._acquire_device()
        self.encoder = self.__init_encoder()

        # Diffusion
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )

        # Decoder for pretrain
        self.denoising_patch_decoder = DenoisingPatchDecoder(
            d_model=args.d_model,
            num_layers=args.d_layers,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
        )

        self.projection = ARFlattenHead(
            d_model=self.d_model,
            patch_len=self.patch_len,
            dropout=args.head_dropout,
        )

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        self.args.device = device
        return device

    def __init_encoder(self):
        """Initialize Qwen2.5-0.5B encoder"""
        encoder = AutoModelForCausalLM.from_pretrained(
            self.args.llm_path,
            output_attentions=True,
            output_hidden_states=True,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        return encoder

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        if self.use_norm:
            # Instance Normalization
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

        # For Qwen2.5-0.5B Encoder
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(x_embedding_bias)

        # Get encoder outputs from Qwen2.5-0.5B
        encoder_outputs = self.encoder(
            inputs_embeds=x_embedding_bias, output_hidden_states=True, return_dict=True
        )
        x_out = encoder_outputs.hidden_states[-1]  # Use last hidden state

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(
            x_patch
        )  # [batch_size * num_features, seq_len, patch_len]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(noise_x_embedding)

        # For Denoising Patch Decoder
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]

        # For Decoder
        predict_x = predict_x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        if self.use_norm:
            predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]
            predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]

        return predict_x

    def forecast_train(self, x, y):
        """Training function for forecasting task
        Args:
            x: look-back window, shape [batch_size, input_len, num_features]
            y: predicted window, shape [batch_size, pred_len, num_features]
        """
        batch_size, input_len, num_features = x.size()

        # Instance Normalization for both x and y if needed
        if self.use_norm:
            # For look-back window
            x_means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - x_means
            x_stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / x_stdevs

        # Process look-back window
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(
            x_embedding_bias
        )  # [batch_size * num_features, seq_len, d_model]

        # Get encoder outputs from look-back window
        encoder_outputs = self.encoder(
            inputs_embeds=x_embedding_bias, output_hidden_states=True, return_dict=True
        )
        x_out = encoder_outputs.hidden_states[-1]  # Use last hidden state as KV

        # x_out: [batch_size * num_features, seq_len, d_model]

        # Process predicted window
        y = self.channel_independence(y)  # [batch_size * num_features, pred_len, 1]
        y_patch = self.patch(y)  # [batch_size * num_features, seq_len_pred, patch_len]

        # Apply diffusion to predicted window patches
        noise_y_patch, noise, t = self.diffusion(
            y_patch
        )  # [batch_size * num_features, seq_len_pred, patch_len]
        noise_y_embedding = self.enc_embedding(
            noise_y_patch
        )  # [batch_size * num_features, seq_len_pred, d_model]
        noise_y_embedding_bias = self.add_sos_token_and_drop_last(
            noise_y_embedding
        )  # [batch_size * num_features, seq_len_pred, d_model]
        noise_y_embedding_bias = self.positional_encoding(
            noise_y_embedding_bias
        )  # [batch_size * num_features, seq_len_pred, d_model]

        # Denoising decoder
        predict_y = self.denoising_patch_decoder(
            query=noise_y_embedding_bias,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=False,
        )  # [batch_size * num_features, seq_len_pred, d_model]

        # Project to original space using ARFlattenHead
        predict_y = predict_y.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len_pred, d_model]
        predict_y = self.projection(
            predict_y
        )  # [batch_size, seq_len_pred * patch_len, num_features]

        # predict_y: [batch_size, pred_len, num_features]

        # Instance Denormalization if needed
        if self.use_norm:
            predict_y = predict_y * x_stdevs[:, 0, :].unsqueeze(1).repeat(
                1, predict_y.size(1), 1
            )
            predict_y = predict_y + x_means[:, 0, :].unsqueeze(1).repeat(
                1, predict_y.size(1), 1
            )

        return predict_y

    def forecasting_test(self, x, max_len):
        """Test function for forecasting task with auto-regressive generation
        Args:
            x: initial look-back window, shape [batch_size, input_len, num_features]
            max_len: maximum length to generate (pred_len)
        """
        batch_size, input_len, num_features = x.size()
        device = x.device

        # Store the original statistics for denormalization
        if self.use_norm:
            x_means = torch.mean(x, dim=1, keepdim=True).detach()
            x_stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = (x - x_means) / x_stdevs

        # Initialize output sequence with look-back window
        generated_sequence = [x]  # [batch_size, input_len, num_features]
        current_input = x  # [batch_size, input_len, num_features]

        # Calculate number of patches needed to generate pred_len
        num_patches_needed = (max_len + self.patch_len - 1) // self.patch_len

        for step in range(num_patches_needed):
            # Process current input through encoder
            current_x = self.channel_independence(
                current_input
            )  # [batch_size * num_features, input_len, 1]
            current_x_patch = self.patch(
                current_x
            )  # [batch_size * num_features, seq_len, patch_len]
            current_x_embedding = self.enc_embedding(
                current_x_patch
            )  # [batch_size * num_features, seq_len, d_model]
            current_x_embedding = self.positional_encoding(
                current_x_embedding
            )  # [batch_size * num_features, seq_len, d_model]

            # Get encoder outputs
            encoder_outputs = self.encoder(
                inputs_embeds=current_x_embedding,
                output_hidden_states=True,
                return_dict=True,
            )
            x_out = encoder_outputs.hidden_states[
                -1
            ]  # [batch_size * num_features, seq_len, d_model]

            # Generate gaussian noise for next step prediction
            noise_shape = (batch_size * num_features, 1, self.patch_len)  # One patch
            gaussian_noise = torch.randn(
                noise_shape, device=device
            )  # [batch_size * num_features, 1, patch_len]

            # Process noise through embedding
            noise_embedding = self.enc_embedding(
                gaussian_noise
            )  # [batch_size * num_features, 1, d_model]
            noise_embedding = self.positional_encoding(
                noise_embedding
            )  # [batch_size * num_features, 1, d_model]

            # Denoising decoder
            predict_patch = self.denoising_patch_decoder(
                query=noise_embedding,
                key=x_out,
                value=x_out,
                is_tgt_mask=True,
                is_src_mask=False,
            )  # [batch_size * num_features, 1, d_model]

            # Project to original space using ARFlattenHead
            predict_patch = predict_patch.reshape(
                batch_size, num_features, -1, self.d_model
            )  # [batch_size, num_features, seq_len_pred, d_model]
            predict_patch = self.projection(
                predict_patch
            )  # [batch_size, seq_len_pred * patch_len, num_features]

            # Take only the last patch_len predictions
            predict_patch = predict_patch[
                :, : self.patch_len, :
            ]  # [batch_size, patch_len, num_features]

            # Denormalize if needed
            if self.use_norm:
                predict_patch = predict_patch * x_stdevs[:, 0, :].unsqueeze(1).repeat(
                    1, predict_patch.size(1), 1
                )
                predict_patch = predict_patch + x_means[:, 0, :].unsqueeze(1).repeat(
                    1, predict_patch.size(1), 1
                )  # [batch_size, patch_len, num_features]

            # Append prediction and update input
            generated_sequence.append(
                predict_patch
            )  # [batch_size, input_len + patch_len, num_features]

            # Update input window by removing oldest patch and adding new prediction
            current_input = torch.cat(
                [current_input[:, self.patch_len :, :], predict_patch], dim=1
            )  # [batch_size, input_len + patch_len, num_features]

        # Concatenate all predictions and trim to exact pred_len
        predictions = torch.cat(
            generated_sequence[1:], dim=1
        )  # [batch_size, input_len + pred_len, num_features]
        return predictions[:, -max_len:, :]  # [batch_size, pred_len, num_features]

    def forward(self, x, y=None):
        if self.task_name == "pretrain":
            return self.pretrain(x)
        elif self.task_name == "finetune":
            if y is not None:  # Training mode
                return self.forecast_train(x, y)
            else:  # Testing mode
                return self.forecasting_test(x, self.pred_len)
        else:
            raise ValueError(f"Task name {self.task_name} not supported")
