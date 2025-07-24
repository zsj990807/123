
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from core.layer.kanlayer import TaylorKANLayer, WaveKANLayer  # 确保正确导入



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class DenseRMoK(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, num_experts=4, drop=0.1, revin_affine=True):
        super(DenseRMoK, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.num_experts = num_experts
        self.drop = drop
        self.revin_affine = revin_affine

        self.gate = nn.Linear(hist_len, num_experts)
        self.experts = nn.ModuleList([
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat"),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat"),
        ])

        self.dropout = nn.Dropout(drop)
        self.rev = RevIN(var_num, affine=revin_affine)

    def forward(self, x):
        # x: [B, L, N]
        x = self.rev(x, 'norm')  # 应用RevIN归一化
        B, L, N = x.shape
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(B * N, L)  # 重塑为[B*N, L]

        score = F.softmax(self.gate(x), dim=-1)  # [B*N, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [B*N, pred_len, num_experts]
        prediction = torch.einsum('blE, bE -> bl', expert_outputs, score)  # 对专家输出进行加权求和

        prediction = prediction.reshape(B, N, -1).permute(0, 2, 1)  # [B, pred_len, N]
        prediction = self.rev(prediction, 'denorm')  # 应用RevIN反归一化
        return prediction


class Model(nn.Module):
    """
    融合了RevIN和DenseRMoK的Transformer模型。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        self.d_model = configs.d_model
        self.var_num = self.enc_in

        # RevIN模块
        self.rev_in = RevIN(self.var_num, affine=True)

        # DenseRMoK模块
        self.dense_rmok = DenseRMoK(
            hist_len=configs.seq_len,
            pred_len=configs.pred_len,
            var_num=self.var_num,
            num_experts=4,
            drop=configs.dropout,
            revin_affine=True
        )

        # Embedding
        self.enc_embedding = DataEmbedding(
            self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # 与DenseRMoK输出结合
        drmok_out = self.dense_rmok(x_enc)
        final_out = dec_out[:, -self.pred_len:, :] + drmok_out

        return final_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out  # [B, L, D]
