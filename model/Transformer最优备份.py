import torch
import torch.nn as nn
from torch_geometric.nn import GATConv  # 需要安装 pytorch_geometric
from torch_geometric.nn import SAGEConv, GINConv

import torch.nn.functional as F

# 定义 GAT 模块
class GATModule(nn.Module):

    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.6):
        super(GATModule, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = self.relu(x)
        return x


class RIMELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_mechanisms=4):
        super(RIMELSTM, self).__init__()
        self.num_mechanisms = num_mechanisms
        self.hidden_size = hidden_size

        # 定义多个LSTM机制
        self.lstm_mechanisms = nn.ModuleList([
            nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True) for _ in range(num_mechanisms)
        ])

        # 门控网络 (Mechanism Selection Network)
        # self.gating_network = nn.Linear(input_size, num_mechanisms)
        self.gating_network = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, num_mechanisms)
        )

    def forward(self, x, hidden_states):
        # x 的维度是 [B, S, D]
        B, S, D = x.size()

        # 使用门控网络生成每个时间步的机制选择权重
        gating_scores = self.gating_network(x).softmax(dim=-1)  # [B, S, num_mechanisms]

        # 初始化输出和新的隐藏状态
        outputs = []
        new_hidden_states = []

        for i, lstm in enumerate(self.lstm_mechanisms):
            # 计算每个机制的输出
            lstm_out, new_hidden = lstm(x, hidden_states[i])
            outputs.append(lstm_out * gating_scores[:, :, i:i + 1])  # 加权机制输出
            new_hidden_states.append(new_hidden)

        # 将所有机制的输出加和作为最终的输出
        final_output = sum(outputs)

        return final_output, new_hidden_states


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


# 首先，确保您已定义了上述的 TCN 模块

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
        # self.gnn = GraphSageModule(in_channels=configs.d_model, out_channels=configs.d_model)
        # self.gnn = GINModule(in_channels=configs.d_model, out_channels=configs.d_model)
        # 添加 LSTM 模块
        # self.lstm = nn.LSTM(input_size=configs.d_model, hidden_size=configs.d_model, num_layers=1, batch_first=True)
        # 使用 RIME 优化 LSTM
        self.rime_lstm = RIMELSTM(input_size=configs.d_model, hidden_size=configs.d_model, num_mechanisms=4)

        # # 融合层（例如，通过加权平均或门控机制）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

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

        # GAT 处理
        B, S, D = enc_out.size()
        enc_out_gat = enc_out.view(B * S, D)  # [B*S, D]
        gat_out = self.gat(enc_out_gat, edge_index)  # [B*S, D]
        gat_out = gat_out.view(B, S, D)  # [B, S, D]

        # LSTM 处理
        # lstm_input = enc_out  # 使用编码器的输出作为 LSTM 的输入
        # lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # [B, S, D]
        # 使用 RIME LSTM 处理
        hidden_states = [None] * self.rime_lstm.num_mechanisms  # 初始化机制的隐藏状态
        rime_out, new_hidden_states = self.rime_lstm(enc_out, hidden_states)
        gating_signal = torch.sigmoid(rime_out)  # [B, S, D]
        optimized_gat_out = gating_signal * gat_out  # [B, S, D]



        # 将优化后的 GAT 输出与编码器输出融合
        enc_out = self.fusion_weight * enc_out + (1 - self.fusion_weight) * optimized_gat_out  # [B, S, D]

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]