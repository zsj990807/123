
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 假设输入特征数为 configs.enc_in，隐藏层维度为 configs.d_model
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.d_model, num_layers=2, batch_first=True)
        self.fc = nn.Linear(configs.d_model, configs.c_out)
        self.pred_len = configs.pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, edge_index=None):
        # x_enc 的形状为 [B, seq_len, enc_in]
        output, _ = self.lstm(x_enc)
        output = self.fc(output)
        # 返回最后 pred_len 个时间步的输出
        return output[:, -self.pred_len:, :]
