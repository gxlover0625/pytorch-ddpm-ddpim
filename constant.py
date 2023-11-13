# 优化器
warmup = 5000

# 训练
batch_size_own = 128
num_workers = 4
lr = 2e-4
total_steps = 800000
grad_clip = 1
ema_decay = 0.9999

# 评估
sample_size = 64
sample_step = 1000
save_step = 5000

# unet模型
T = 1000
ch = 128
ch_mult = [1, 2, 2, 2]
attn = [1]
num_res_blocks = 2
dropout = 0.1

# trainer
beta_1 = 1e-4
beta_T = 0.02

# sampler
img_size = 32
mean_type = 'epsilon'
var_type = 'fixedlarge'
logdir = './logs/DDPM_CIFAR10_EPS'
