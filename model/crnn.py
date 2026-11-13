import math
import torch
import torch.nn as nn


class DynamicSelfAttention(nn.Module):
    """Multihead Self-Attention that computes sequence length dynamically for ONNX export."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()

        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(attn_out)


class CRNN(nn.Module):

    def __init__(self, input_height, num_classes):
        super().__init__()
        self.filter_size = 64
        self.image_size = 1  # input_height // input_height
        self.hidden_size = 256
        self.lstm_hidden = 256
        self.lstm_layer = 2
        self.attention_head = 8

        # --- CNN Backbone ---
        self.conv1 = nn.Conv2d(3, self.filter_size, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.bn12 = nn.BatchNorm2d(self.filter_size)

        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 1, 1)
        self.conv22 = nn.Conv2d(
            self.filter_size * 2, self.filter_size * 2, 3, 1, 1
        )
        self.bn2 = nn.BatchNorm2d(self.filter_size * 2)
        self.bn22 = nn.BatchNorm2d(self.filter_size * 2)

        self.conv23 = nn.Conv2d(
            self.filter_size * 2, self.filter_size * 4, 3, 1, 1
        )
        self.bn23 = nn.BatchNorm2d(self.filter_size * 4)

        self.conv24 = nn.Conv2d(
            self.filter_size * 4, self.filter_size * 4, 3, 1, 1
        )
        self.bn24 = nn.BatchNorm2d(self.filter_size * 4)

        self.conv25 = nn.Conv2d(
            self.filter_size * 4, self.filter_size * 4, 4, 1, 0
        )
        self.bn25 = nn.BatchNorm2d(self.filter_size * 4)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(
            self.filter_size * 4 * self.image_size,
            self.hidden_size,
            kernel_size=1,
        )

        # --- Sequence Modeling ---
        self.lstm1 = nn.LSTM(
            self.hidden_size,
            self.lstm_hidden,
            num_layers=self.lstm_layer,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

        self.attention = DynamicSelfAttention(
            embed_dim=self.lstm_hidden * 2,
            num_heads=self.attention_head,
            dropout=0.1,
        )

        self.lstm2 = nn.LSTM(
            self.lstm_hidden * 2,
            self.lstm_hidden,
            num_layers=self.lstm_layer,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

        self.layer_norm1 = nn.LayerNorm(self.lstm_hidden * 2)
        self.layer_norm2 = nn.LayerNorm(self.lstm_hidden * 2)
        self.layer_norm3 = nn.LayerNorm(self.lstm_hidden * 2)

        self.fc2 = nn.Linear(self.lstm_hidden * 2, num_classes)

    def forward(self, x):
        # --- CNN Feature Extraction ---
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn12(self.conv12(output)))
        output = self.pool(output)

        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn22(self.conv22(output)))
        output = self.pool(output)

        output = self.relu(self.bn23(self.conv23(output)))
        output = self.relu(self.bn24(self.conv24(output)))
        output = self.pool(output)

        output = self.relu(self.bn25(self.conv25(output)))

        bs, c, h, w = output.size()
        output = output.permute(0, 1, 3, 2)
        output = output.reshape(bs, c * h, w)

        output = self.conv3(output)
        output = output.permute(0, 2, 1)  # (B, T, C)

        # --- BiLSTM 1 ---
        output1, _ = self.lstm1(output)
        output1 = self.layer_norm1(output1)

        # --- Self-Attention Block with Residual & Norm ---
        attn_out = self.attention(output1)
        attn_out = self.layer_norm2(output1 + self.dropout(attn_out))

        # --- BiLSTM 2 ---
        output2, _ = self.lstm2(attn_out)
        output2 = self.layer_norm3(output2)

        # --- Final Dense Projection ---
        output = output1 + attn_out + output2
        output = self.fc2(output)
        return output