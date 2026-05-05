import torch
import torch.nn as nn
import torch.nn.functional as F


def build_periodic_adj(Q_cal, L=0, H=0):
    """
    Q_cal: (B, Q, N)
    Returns:
        x_adj: (B, L, N)
        y_adj: (B, H, N)
    """
    B, Q, N = Q_cal.shape
    total_len = L + H

    # The number of repetitions required (rounded up to the nearest integer)
    repeat_times = (total_len + Q - 1) // Q

    # Periodic Expansion
    Q_tiled = Q_cal.repeat(1, repeat_times, 1)  # (B, repeat_times * Q, N)

    # Truncate
    x_adj = Q_tiled[:, :L, :]
    y_adj = Q_tiled[:, L:L + H, :]

    return x_adj, y_adj


class SeasonCalibrator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(self.args.seq_len * 2, self.args.d_ff),
            nn.GELU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.d_ff, self.args.cycle)
        )

    def forward(self, x, Q):
        return self.mlp(torch.cat((Q, x), dim=1).permute(0, 2, 1)).permute(0, 2, 1)



class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size

    def forward(self, index, length, cycle_data):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return cycle_data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.args = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.use_revin = configs.use_revin
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']

        self.input_dim = self.seq_len
        self.output_dim = self.pred_len
        if self.model_type == 'linear':
            self.model = nn.Linear(self.input_dim, self.output_dim)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.d_ff),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.d_ff, self.output_dim),
            )
        self.calibrator = SeasonCalibrator(configs)

    def forward(self, x, cycle_index, cycle_data):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        batch_size, _, _ = x.shape
        cycle_data = torch.Tensor(cycle_data).to(cycle_index.device)

        seq_mean, seq_var = torch.zeros_like(x), torch.ones_like(x)
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        x_norm = x

        # remove the cycle of the input data
        Q_lookback = self.cycleQueue(cycle_index, self.seq_len, cycle_data)
        x_deseason = x - Q_lookback

        Q_cal = self.calibrator(x_deseason, Q_lookback)
        x_Q_adj, y_Q_adj = build_periodic_adj(Q_cal, self.seq_len, self.pred_len)
        x_deseason = x_deseason - x_Q_adj
        y_deseason = self.model(x_deseason.permute(0, 2, 1)).permute(0, 2, 1)  # batch_size, pred_len, N
        Q_horizon = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len, cycle_data)
        y = y_deseason + Q_horizon + y_Q_adj
        y_norm = y

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y, x_deseason, y_deseason, Q_lookback, Q_horizon, x_norm, y_norm, seq_mean, seq_var, x_Q_adj, y_Q_adj
