
from torch_geometric.nn import GATConv  # 需要安装 pytorch_geometric
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn as nn

class LSTMGatedFusion(nn.Module):
    def __init__(self, d_model, hidden_size=None, num_layers=1, bidirectional=False):
        """
        d_model: Transformer 和 GAT 模块输出的特征维度
        hidden_size: LSTM 隐藏层的维度（默认等于 d_model）
        num_layers: LSTM 层数
        bidirectional: 是否使用双向 LSTM
        """
        super(LSTMGatedFusion, self).__init__()
        if hidden_size is None:
            hidden_size = d_model
        self.lstm = nn.LSTM(
            input_size=2 * d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        final_dim = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(final_dim, d_model)

    def forward(self, x_trans, x_gat):
        """
        x_trans, x_gat: [B, S, D]
        输出: [B, S, D] 融合后的特征
        """
        # 拼接两个模块的输出: [B, S, 2D]
        x_cat = torch.cat([x_trans, x_gat], dim=-1)
        # 输入 LSTM，获得 LSTM 输出: [B, S, hidden] 或 [B, S, hidden*2] (如果双向)
        lstm_out, _ = self.lstm(x_cat)
        # 通过线性层映射到 D 维度，并使用 Sigmoid 获得门控权重: [B, S, D]
        gate = torch.sigmoid(self.linear(lstm_out))
        # 融合两部分输出
        fused_output = gate * x_trans + (1 - gate) * x_gat
        return fused_output


# # 自定义模块
# class FourierEnhancer(nn.Module):
#     def __init__(self, keep_ratio=0.3, mode='lowpass'):
#         super().__init__()
#         self.keep_ratio = keep_ratio
#         self.mode = mode
#
#     def forward(self, x):
#         B, S, D = x.shape
#         x_fft = torch.fft.rfft(x, dim=1)
#         F = x_fft.size(1)
#         cutoff = int(F * self.keep_ratio)
#
#         x_fft_filtered = x_fft.clone()
#         if self.mode == 'lowpass':
#             x_fft_filtered[:, cutoff:] = 0
#         elif self.mode == 'highpass':
#             x_fft_filtered[:, :cutoff] = 0
#         else:
#             raise ValueError("mode must be 'lowpass' or 'highpass'")
#         return torch.fft.irfft(x_fft_filtered, n=S, dim=1)


# 定义 GAT 模块
class GATModule(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, dropout=0.6):
        super(GATModule, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        edge_weight = self.compute_edge_weights(x, edge_index)
        x = self.gat_conv(x, edge_index, edge_attr=edge_weight)
        x = self.relu(x)
        return x

    def compute_edge_weights(self, x, edge_index):
        """
        根据节点特征计算边权重，可以使用学习的边权重。
        """
        edge_weight = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=-1)
        edge_weight = torch.exp(-edge_weight)  # 使用高斯核函数进行归一化
        return edge_weight


class Model(nn.Module):
    """
    Transformer with added LSTM and GAT modules for multivariate time series classification
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

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 添加 GAT 模块
        self.gat = GATModule(in_channels=configs.d_model, out_channels=configs.d_model, heads=1, concat=False)
        # self.fourier = FourierEnhancer(keep_ratio=0.3, mode='lowpass')
        # 添加 GAT 模块
        self.gat = GATModule(in_channels=configs.d_model, out_channels=configs.d_model, heads=1, concat=False)

        # 在这里添加正则化层：
        self.gat_norm = nn.LayerNorm(configs.d_model)
        # self.gat_dropout = nn.Dropout(configs.dropout)
        self.fusion_norm = nn.LayerNorm(configs.d_model)
        # self.fusion_dropout = nn.Dropout(configs.dropout)

        # 使用lstm实现门控融合
        self.lstm_gated_fusion = LSTMGatedFusion(d_model=configs.d_model, bidirectional=False)
        # self.grn_gated_fusion = GRNGatedFusion(d_model=configs.d_model)

        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, S, D]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B, S, D]

        # 🚨 暂时注释掉所有复杂组件
        # # GAT 处理
        # B, S, D = enc_out.size()
        # enc_out_gat = enc_out.reshape(B * S, D)  # [B*S, D]
        # gat_out = self.gat(enc_out_gat, edge_index)
        # gat_out = gat_out.view(B, S, D)
        # gat_out = self.gat_norm(gat_out)  # 新增
        # gat_out = self.gat_dropout(gat_out)  # 新增
        #
        # fused_enc_out = self.lstm_gated_fusion(enc_out, gat_out)
        # fused_enc_out = self.fusion_norm(fused_enc_out)  # 新增
        # fused_enc_out = self.fusion_dropout(fused_enc_out)  # 新增

        # 🎯 直接使用Transformer编码器的输出
        fused_enc_out = enc_out

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, fused_enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]