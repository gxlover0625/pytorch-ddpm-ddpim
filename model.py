import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super(TimeEmbedding, self).__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)  # [d_model // 2]
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]  # [T,d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # [T,d_model//2,2]
        emb = emb.view(T, d_model)
        # 默认冻结,不进行梯度更新
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # [batch_size,d_model]
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)  # [batch_size,dim]
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)  # [batch_size,dim]
        return emb  # [batch_size,dim]


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super(DownSample, self).__init__()
        # H -> [(H+1)/2]下, W -> [(W+1)/2]下
        # kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x  # [B,in_ch,H',W']


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super(UpSample, self).__init__()
        # H -> H, W -> W
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # H -> 2H, W -> 2W
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        # 尺寸不变
        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    # 要满足x的维度C与in_ch一致
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)  # [B,C,H,W]

        # [B,H,W,C]
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # [B,HW,C]
        k = k.view(B, C, H * W)  # [B,C,HW]
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)  # [B,HW,HW]

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)  # [B,HW,C]
        h = torch.bmm(w, v)  # [B,HW,C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        h = self.proj(h)  # [B,C,H,W]

        return x + h  # [B,C,H,W]


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(
            # 总共分为32组,每组 in_ch // 32个通道
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            # 不改变尺寸
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)  # [B,C,H,W]
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        # 要满足x的通道数与in_ch一致
        # [B,in_ch,H,W] -> [B,out_ch,H,W]
        h = self.block1(x)
        # temb的维度[batch_size,4 * ch] -> [B,out_ch] -> [B,out_ch,H,W]
        h += self.temb_proj(temb)[:, :, None, None]  # [B,out_ch,H,W]
        h = self.block2(h)  # [B,out_ch,H,W]

        h = h + self.shortcut(x)  # x[B,in_ch,H,W]->[B,out_ch,H,W]
        h = self.attn(h)  # [B,out_ch,H,W]
        return h


class UNet(nn.Module):
    # ch_mult通道数的倍数列表[1,2,2,2]
    # T = 1000
    # ch = 128,初识通道数
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super(UNet, self).__init__()
        tdim = ch * 4

        self.time_embedding = TimeEmbedding(T, ch, tdim)  # ch是d_model,tdim是输出维度
        # 图片尺寸不变,通道数变为128
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # [128]
        now_ch = ch  # 128

        # chs [128 ,128 ,128 ,128 ,256 ,256 ,256, 256 ,256 ,256 ,256 ,256]
        #     [init,Res1,Res2,Down,Res1,Res2,Down,Res1,Res2,Down,Res1,Res2]
        #     [32  ,32,  32  ,16  ,16  ,16  ,8   ,8   ,8   ,4   ,4   ,4]
        # 4层Res,3层Down
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    # 不改变图片尺寸
                    ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                # 图片尺寸减半,通道不变
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # chs [256,256]
        #     [mid,mid]
        #     [4  ,4]
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False)
        ])

        self.upblocks = nn.ModuleList()
        # chs [128 ,128 ,128 ,128 ,256 ,256 ,256, 256 ,256 ,256 ,256 ,256]
        #     [init,Res1,Res2,Down,Res1,Res2,Down,Res1,Res2,Down,Res1,Res2]
        #     [32  ,32,  32  ,16  ,16  ,16  ,8   ,8   ,8   ,4   ,4   ,4]

        # [2,2,2,1]
        # [512 ,256 ]
        # [init,Res1]
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):  # 2层block
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, 3, kernel_size=3, stride=1, padding=1)
        )

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # t的维度[batch_size] -> # [batch_size,4 * ch]
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            # print(h.shape)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        # [B,128,H,W] -> [B,3,H,W]
        h = self.tail(h)
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
