import json
import torch
import os
import numpy as np
from model import VAE
from data_loader import MusicArrayLoader
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


# some initialization
with open('model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('params'):
    os.mkdir('params')
save_path = 'params/{}.pt'.format(args['name'])
writer = SummaryWriter('log/{}'.format(args['name']))
model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'],
            args['rhythm_dim'], args['time_step'])
if args['if_parallel']:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
optimizer = optim.Adam(model.parameters(), lr=args['lr'])
if args['decay'] > 0:
    scheduler = MinExponentialLR(optimizer, gamma=args['decay'], minimum=1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')
step, pre_epoch = 0, 0
model.train()
dl = MusicArrayLoader(args['data_path'], args['time_step'], 16)
dl.chunking()

# end of initialization


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  distribution_1,
                  distribution_2,
                  step,
                  beta=.1):
    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction='elementwise_mean')
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='elementwise_mean')
    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    return CE1 + CE2 + beta * (KLD1 + KLD2)


def train(step):
    batch, c = dl.get_batch(args['batch_size'])
    print(batch.shape, c.shape)
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()
    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()
    optimizer.zero_grad()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)
    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=args['beta'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1
    print('batch loss: {:.5f}'.format(loss.item()))
    writer.add_scalar('batch_loss', loss.item(), step)
    if args['decay'] > 0:
        scheduler.step()
    dl.shuffle_samples()
    return step


while dl.get_n_epoch() < args['n_epochs']:
    step = train(step)
    if dl.get_n_epoch() != pre_epoch:
        pre_epoch = dl.get_n_epoch()
        torch.save(model.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            model.cuda()
        print('Model saved!')
