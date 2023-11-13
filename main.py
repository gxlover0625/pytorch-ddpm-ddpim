import copy
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from constant import *
from tqdm import trange
from model import UNet
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, DDIM

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, warmup) / warmup


def train():
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=batch_size_own, shuffle=True,
        num_workers=num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)  # 生成迭代器

    net_model = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ema_model = copy.deepcopy(net_model)

    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(net_model, beta_1, beta_T, T).to(device)
    ema_sampler = GaussianDiffusionSampler(ema_model, beta_1, beta_T, T, img_size, mean_type, var_type).to(device)

    os.makedirs(os.path.join(logdir, 'sample'))
    x_T = torch.randn(sample_size, 3, img_size, img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:sample_size]) + 1) / 2
    writer = SummaryWriter(logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), grad_clip)
            optim.step()
            sched.step()

            ema(net_model, ema_model, ema_decay)  # 更新参数
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            if sample_step > 0 and step % sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1.0) / 2
                    path = os.path.join(logdir, 'sample', '%d.png' % (step + 84000))
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if save_step > 0 and step % save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(logdir, 'ckpt_%d.pt' % (step + 84000)))
                writer.close()


def eval(step=save_step):
    model = UNet(T=T, ch=ch, ch_mult=ch_mult, attn=attn, num_res_blocks=num_res_blocks, dropout=dropout)
    sampler = DDIM(model, beta_1, beta_T, T, img_size=img_size, mean_type=mean_type,
                   var_type=var_type).to(device)
    ckpt = torch.load(os.path.join(logdir, 'ckpt_%d.pt' % step))
    model.load_state_dict(ckpt['net_model'])
    model.eval()
    with torch.no_grad():
        x_T = torch.randn(sample_size, 3, img_size, img_size)
        x_T = x_T.to(device)
        x_0, imgs = sampler(x_T, process=True, sample_steps=5)
        grid = (make_grid(x_0) + 1.0) / 2
        B, T_2, C, H, W = imgs.shape

        for i in range(B):
            grid_cur = (make_grid(imgs[i], nrow=20) + 1) / 2
            cur_path = os.path.join(logdir, 'sample_%d.png' % i)
            save_image(grid_cur, cur_path)
        path = os.path.join(logdir, 'sample.png')
        save_image(grid, path)
    model.train()


train()
# eval(step=xx)
