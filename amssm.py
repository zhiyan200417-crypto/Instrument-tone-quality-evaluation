import torch
import torch.nn as nn
import torch.nn.functional as F

from models.asma_crnn_model import SimplifiedBiGSSM1D


class AF_ASMA_CRNN_ADSR_Encoder(nn.Module):
    """

    参数
    ----
    n_mels            : int        — Mel 滤波器组数
    cnn_channels      : list[int]  — CNN 各块输出通道数
    cnn_kernel_size   : int        — CNN 卷积核大小
    rnn_input_dim     : int        — 共享投影层输出维度
    rnn_hidden_size   : int        — BiGRU 每方向隐藏单元数
    rnn_layers        : int        — BiGRU 层数
    d_aux             : int        — SSM 富化分支瓶颈维度
    d_state           : int        — SSM 隐状态维度
    gate_init         : float      — fusion_gate 初始值
    router_reduction  : int        — Router MLP 中间维度
    aux_zero_prob     : float      — DropPath 概率
    dropout           : float      — 通用 Dropout 概率
    aux_dropout       : float      — SSM 富化分支 Dropout 概率
    a_init_start      : float      — SSM A_param linspace 起始值
    a_init_end        : float      — SSM A_param linspace 结束值
    n_heads           : int        — Cross-Attention 注意力头数
    attn_dropout      : float      — Cross-Attention 和 FFN 的 Dropout
    acoustic_dim      : int        — 声学描述符维度 K

    示例
    ----
    >>> encoder = AF_ASMA_CRNN_ADSR_Encoder(n_mels=80, acoustic_dim=16)
    >>> mel_ad = torch.randn(4, 18, 80)
    >>> mel_s  = torch.randn(4, 210, 80)
    >>> mel_r  = torch.randn(4, 26, 80)
    >>> ac_ad  = torch.randn(4, 18, 16)
    >>> ac_s   = torch.randn(4, 210, 16)
    >>> ac_r   = torch.randn(4, 26, 16)
    >>> out = encoder(mel_ad, mel_s, mel_r, ac_ad, ac_s, ac_r)
    >>> out.shape
    torch.Size([4, 256])
    """

    is_adsr_model = True
    has_aux_output = False

    def __init__(
        self,
        n_mels=80,
        cnn_channels=None,
        cnn_kernel_size=3,
        rnn_input_dim=256,
        rnn_hidden_size=128,
        rnn_layers=2,
        d_aux=64,
        d_state=4,
        gate_init=-1.5,
        router_reduction=32,
        aux_zero_prob=0.2,
        dropout=0.2,
        aux_dropout=0.4,
        a_init_start=0.5,
        a_init_end=2.5,
        n_heads=4,
        attn_dropout=0.1,
        acoustic_dim=16,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.dropout_p = dropout
        self.aux_zero_prob = aux_zero_prob
        self.acoustic_dim = acoustic_dim

        # ================================================================ #
        #  共享 CNN 卷积块                                                   #
        # ================================================================ #
        conv_blocks = []
        in_ch = 1
        for out_ch in cnn_channels:
            conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, cnn_kernel_size,
                          padding=cnn_kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # 段特异性池化配置
        self.pool_sizes_short = [(1, 2), (1, 2), (1, 2)]  # AD/R 段
        self.pool_sizes_long = [(2, 2), (2, 2), (1, 2)]   # S 段

        # CNN 输出的频率维度
        freq_after_cnn = n_mels
        for _, fp in self.pool_sizes_short:
            freq_after_cnn //= fp
        cnn_output_dim = cnn_channels[-1] * freq_after_cnn

        # ================================================================ #
        #  共享线性投影层                                                     #
        # ================================================================ #
        self.projection = nn.Linear(cnn_output_dim, rnn_input_dim)

        # ================================================================ #
        #  段类型嵌入：AD=0, S=1, R=2                                       #
        # ================================================================ #
        self.segment_type_embed = nn.Embedding(3, rnn_input_dim)

        # ================================================================ #
        #  SSM 富化分支                                                      #
        # ================================================================ #
        self.aux_down_proj = nn.Linear(rnn_input_dim, d_aux)
        self.aux_norm = nn.LayerNorm(d_aux)
        self.aux_bigssm = SimplifiedBiGSSM1D(
            dim=d_aux, d_state=d_state, dropout=aux_dropout,
            a_init_start=a_init_start, a_init_end=a_init_end,
        )
        self.aux_up_proj = nn.Linear(d_aux, rnn_input_dim, bias=False)

        # ================================================================ #
        #  Content-Adaptive Router                                          #
        # ================================================================ #
        router_in_dim = rnn_input_dim * 2
        self.router_down = nn.Linear(router_in_dim, router_reduction)
        self.router_up = nn.Linear(router_reduction, rnn_input_dim)

        # ================================================================ #
        #  融合门控                                                          #
        # ================================================================ #
        self.fusion_gate = nn.Parameter(
            torch.ones(rnn_input_dim) * gate_init
        )

        # ================================================================ #
        #  融合后归一化层                                                     #
        # ================================================================ #
        self.enrichment_norm = nn.LayerNorm(rnn_input_dim)

        # ================================================================ #
        #  帧级跨段注意力融合（分辨率协调器）                                  #
        # ================================================================ #
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=rnn_input_dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm1 = nn.LayerNorm(rnn_input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(rnn_input_dim, rnn_input_dim * 2),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(rnn_input_dim * 2, rnn_input_dim),
        )
        self.attn_norm2 = nn.LayerNorm(rnn_input_dim)

        # ================================================================ #
        #  声学描述符归一化 + 投影 + 全局门控                                 #
        #  ─────────────────────────────────                                #
        #  注入位置：Cross-Attention+FFN 之后、BiGRU 之前。                   #
        #  三个组件的名字都含 "acoustic"，这是 train.py 的 warm-start        #
        #  键过滤与两阶段微调参数分组所依赖的约定，不可改名。                    #
        # ================================================================ #
        self.acoustic_norm = nn.LayerNorm(acoustic_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, rnn_input_dim)
        self.acoustic_gate = nn.Parameter(torch.tensor(-2.0))

        # ================================================================ #
        #  主时序建模：2 层 BiGRU                                            #
        # ================================================================ #
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        self.embedding_dim = 2 * rnn_hidden_size

        # ================================================================ #
        #  参数初始化                                                        #
        # ================================================================ #
        self.apply(self._init_weights)
        self.aux_bigssm.gssm_fwd._init_gates()
        self.aux_bigssm.gssm_bwd._init_gates()

        # acoustic_proj 使用较小方差初始化（配合 sigmoid(-2)≈0.12 的门控，
        # 确保初始声学贡献较小，不破坏 warm-start backbone 的特征分布）
        nn.init.normal_(self.acoustic_proj.weight, std=0.01)
        nn.init.zeros_(self.acoustic_proj.bias)

    @staticmethod
    def _init_weights(module):
        """参数初始化策略（与 ASMA_CRNN_ADSR_Encoder 完全一致）。"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                    nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                    nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _process_cnn(self, mel, pool_sizes):
        """共享 CNN 前端 + 段特异性池化 + 共享线性投影。"""
        x = mel.unsqueeze(1)  # (B, 1, T, F)

        for conv_block, (tp, fp) in zip(self.conv_blocks, pool_sizes):
            x = conv_block(x)
            x = F.max_pool2d(x, (tp, fp))
            x = F.dropout2d(x, p=self.dropout_p, training=self.training)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.projection(x)

        return x

    def _drop_path(self, h_ssm):
        """Per-sample 随机置零（与原版 ASMA-CRNN 一致）。"""
        if not self.training or self.aux_zero_prob <= 0.0:
            return h_ssm

        keep_prob = 1.0 - self.aux_zero_prob
        mask = (torch.rand(h_ssm.size(0), 1, 1, device=h_ssm.device,
                           dtype=h_ssm.dtype) >= self.aux_zero_prob).float()
        return h_ssm * mask / keep_prob

    def forward(self, mel_ad, mel_s, mel_r, ac_ad, ac_s, ac_r):
        """前向传播：三阶段建模 + 声学注入。

        参数
        ----
        mel_ad : (B, T_ad, n_mels)  — AD 段 Mel 谱图
        mel_s  : (B, T_s,  n_mels)  — S 段 Mel 谱图
        mel_r  : (B, T_r,  n_mels)  — R 段 Mel 谱图
        ac_ad  : (B, T_ad, K)       — AD 段声学描述符
        ac_s   : (B, T_s,  K)       — S 段声学描述符
        ac_r   : (B, T_r,  K)       — R 段声学描述符

        返回
        ----
        (B, embedding_dim) — 嵌入向量，embedding_dim = 2 × rnn_hidden_size
        """
        # ---- ADSR 前端：共享 CNN（段特异性池化）+ 投影 + 段类型嵌入 ----
        x_ad = self._process_cnn(mel_ad, self.pool_sizes_short)  # (B, t_ad, 256)
        x_s = self._process_cnn(mel_s, self.pool_sizes_long)     # (B, t_s, 256)
        x_r = self._process_cnn(mel_r, self.pool_sizes_short)    # (B, t_r, 256)

        t_ad = x_ad.shape[1]
        t_s = x_s.shape[1]
        t_r = x_r.shape[1]

        x_ad = x_ad + self.segment_type_embed.weight[0]
        x_s = x_s + self.segment_type_embed.weight[1]
        x_r = x_r + self.segment_type_embed.weight[2]

        # ---- 帧级拼接 ----
        x_proj = torch.cat([x_ad, x_s, x_r], dim=1)             # (B, T_total, 256)

        # ---- 第一阶段：SSM 富化 + 自适应门控 ----
        h_aux = self.aux_down_proj(x_proj)
        h_aux = self.aux_norm(h_aux)
        h_aux = self.aux_bigssm(h_aux)
        h_ssm = self.aux_up_proj(h_aux)

        h_ssm = self._drop_path(h_ssm)

        x_mean = x_proj.mean(dim=1)
        x_std = x_proj.std(dim=1)
        router_in = torch.cat([x_mean, x_std], dim=-1)

        gate_mod = torch.sigmoid(
            self.router_up(F.relu(self.router_down(router_in)))
        )

        gate = torch.sigmoid(self.fusion_gate) * gate_mod
        gate = gate.unsqueeze(1)

        x_enriched = x_proj + gate * h_ssm
        x_enriched = self.enrichment_norm(x_enriched)

        # ---- 第二阶段：Cross-Attention + FFN（分辨率协调） ----
        attn_out, _ = self.cross_attn(
            x_enriched, x_enriched, x_enriched)
        x_attn = self.attn_norm1(x_enriched + attn_out)

        ffn_out = self.ffn(x_attn)
        x_attn = self.attn_norm2(x_attn + ffn_out)

        # ---- 声学描述符注入（分段对齐 + 门控加法） ----
        # 各段声学通过 adaptive_avg_pool1d 对齐到 CNN 输出帧数
        ac_ad_aligned = F.adaptive_avg_pool1d(
            ac_ad.permute(0, 2, 1), t_ad
        ).permute(0, 2, 1)                                       # (B, t_ad, K)

        ac_s_aligned = F.adaptive_avg_pool1d(
            ac_s.permute(0, 2, 1), t_s
        ).permute(0, 2, 1)                                       # (B, t_s, K)

        ac_r_aligned = F.adaptive_avg_pool1d(
            ac_r.permute(0, 2, 1), t_r
        ).permute(0, 2, 1)                                       # (B, t_r, K)

        # 拼接为与 Mel 特征相同的时间维
        ac_cat = torch.cat(
            [ac_ad_aligned, ac_s_aligned, ac_r_aligned], dim=1
        )                                                        # (B, T_total, K)

        ac_cat = self.acoustic_norm(ac_cat)
        ac_proj = self.acoustic_proj(ac_cat)                     # (B, T_total, 256)

        x_attn = x_attn + torch.sigmoid(self.acoustic_gate) * ac_proj

        # ---- 第三阶段：BiGRU 主时序建模 ----
        h_out, _ = self.rnn(x_attn)                              # (B, T_total, 256)

        # ---- 时间均值池化 ----
        return h_out.mean(dim=1)                                 # (B, 256)

    def get_gate_stats(self):
        """提取融合门控、SSM 记忆衰减、段类型嵌入和声学门控的统计值。

        用于训练过程中的权重日志记录。
        额外返回 acoustic_gate 标量门控的 sigmoid 值，反映声学描述符注入强度
        的学习演化。train.py 的 CSV 日志会自动检测并记录该字段。

        返回
        ----
        dict : 包含以下键值对：
            gate           — σ(fusion_gate).mean()
            A              — σ(A_param).mean()（前后向平均）
            embed_ad       — 段类型嵌入 AD 向量的 L2 范数
            embed_s        — 段类型嵌入 S 向量的 L2 范数
            embed_r        — 段类型嵌入 R 向量的 L2 范数
            acoustic_gate  — σ(acoustic_gate)，声学注入门控强度
        """
        with torch.no_grad():
            stats = {}

            stats["gate"] = torch.sigmoid(self.fusion_gate).mean().item()

            a_fwd = torch.sigmoid(
                self.aux_bigssm.gssm_fwd.A_param).mean().item()
            a_bwd = torch.sigmoid(
                self.aux_bigssm.gssm_bwd.A_param).mean().item()
            stats["A"] = (a_fwd + a_bwd) / 2.0

            stats["embed_ad"] = self.segment_type_embed.weight[0].norm().item()
            stats["embed_s"] = self.segment_type_embed.weight[1].norm().item()
            stats["embed_r"] = self.segment_type_embed.weight[2].norm().item()

            # AF 模型独有：声学描述符注入门控强度
            stats["acoustic_gate"] = torch.sigmoid(self.acoustic_gate).item()

        return stats
