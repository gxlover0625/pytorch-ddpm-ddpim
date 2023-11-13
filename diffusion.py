import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v, t, x_shape):
    # V[t]系数
    # t[B] x_shape [B,C,H,W]
    out = torch.gather(v, index=t, dim=0).float()
    # [B,1,1,1]
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# 前向
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super(GaussianDiffusionTrainer, self).__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        # x [B,C,H,W]
        # t [B], [T1,T2,T3,...,TB]
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)  #
        # 加噪
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32, mean_type='eps', var_type='fixedlarge'):
        super(GaussianDiffusionSampler, self).__init__()
        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        # beta_1,...,beta_T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # alpha_1,...,alpha_T
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 在最前面填充了一个1元素
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # 猜测正态分布的方差不能为0,因为方差出现在系数的分母上
        self.register_buffer('posterior_log_var_clipped',
                             torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 根据x_t去计算x_0
    # torch.sqrt(1. / alphas_bar - 1)
    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    # 计算p(x_{t-1}|x_t)的均值和方差
    def q_mean_variance(self, x_0, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def p_mean_variance(self, x_t, t):
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        if self.mean_type == 'epsilon':
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)
        return model_mean, model_log_var

    def forward(self, x_T, process=False):
        x_t = x_T
        imgs = [torch.clip(x_t, -1, 1).cpu()]
        count_step = 1
        for time_step in tqdm(reversed(range(self.T)), desc="Inference"):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t, t)  # 计算方差和均值
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
            if count_step % 50 == 0:
                imgs.append(torch.clip(x_t, -1, 1).cpu())
            count_step += 1
        x_0 = x_t
        if process:
            imgs = torch.stack(imgs, dim=1)
            return torch.clip(x_0, -1, 1), imgs
            # return x_0
        else:
            return torch.clip(x_0, -1, 1)


class DDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32, mean_type='eps', var_type='fixedlarge',
                 clip_denoised=True, ddim_eta=0.0):
        super(DDIM, self).__init__()
        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.clip_denoised = clip_denoised
        self.ddim_eta = ddim_eta

        # beta_1,...,beta_T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # alpha_1,...,alpha_T
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 在最前面填充了一个1元素
        self.register_buffer('alphas_bar_prev_whole', F.pad(alphas_bar, [1, 0], value=1))
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # 猜测正态分布的方差不能为0,因为方差出现在系数的分母上
        self.register_buffer('posterior_log_var_clipped',
                             torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 根据x_t去计算x_0
    # torch.sqrt(1. / alphas_bar - 1)
    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def forward(self, x_T, sample_steps=10, process=False):
        assert self.T % sample_steps == 0
        x_t = x_T
        imgs = [torch.clip(x_t, -1, 1).cpu()]
        # [100,80,60,40,20]
        t_seq = torch.arange(sample_steps, self.T + 1, sample_steps)
        # [80,60,40,20,0]
        t_prev_seq = t_seq - sample_steps
        for i, j in tqdm(zip(reversed(list(t_seq)), reversed(list(t_prev_seq))), desc='Inference'):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * i
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * j
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_bar_prev_whole, t, x_t.shape)
            alpha_cumprod_t_prev = extract(self.alphas_bar_prev_whole, prev_t, x_t.shape)

            # 2. predict noise using model
            eps = self.model(x_t, t - 1)

            # 3. get the predicted x_0
            x_0 = self.predict_xstart_from_eps(x_t, t - 1, eps)
            if self.clip_denoised:
                x_0 = torch.clamp(x_0, min=-1., max=1.)

            # 4. compute variance:
            sigma_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * eps
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * x_0 + pred_dir_xt + sigma_t ** 2 * torch.randn_like(x_t)
            x_t = x_prev
            imgs.append(torch.clip(x_t, -1, 1).cpu())
        if process:
            imgs = torch.stack(imgs, dim=1)
            return torch.clip(x_t, -1, 1), imgs
        else:
            return torch.clip(x_t, -1, 1)
