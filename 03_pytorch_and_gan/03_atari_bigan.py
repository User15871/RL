# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 21:35:16 2021

@author: andy
"""

import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 32

# dimension input image will be rescaled
IMAGE_SIZE = 128 #Only accept 8*2**n
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000
CONV_STACK_NUM = 0
while 2**CONV_STACK_NUM*16 <= IMAGE_SIZE:
    CONV_STACK_NUM += 1


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype = np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipein_x = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            )
        
        self.conv_pipelist_x = nn.ModuleList(
                [nn.Sequential(
                nn.Conv2d(in_channels=2**i*DISCR_FILTERS, out_channels=2**(i+1)*DISCR_FILTERS,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(2**(i+1)*DISCR_FILTERS),
                nn.ReLU()
                ) for i in range(CONV_STACK_NUM)]
            )
        
        
        self.conv_pipein_z = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=2**CONV_STACK_NUM*GENER_FILTERS,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d (2**CONV_STACK_NUM*GENER_FILTERS),
            nn.ReLU()
            )
        
        
        
        self.conv_pipeout_xz = nn.Sequential(
            nn.Conv2d(in_channels=2**CONV_STACK_NUM*(DISCR_FILTERS+GENER_FILTERS), out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x = self.conv_pipein_x(x)
        for conv_pipe_x in self.conv_pipelist_x:
            x = conv_pipe_x(x)
        z = self.conv_pipein_z(z)
        xz = torch.cat((x,z), 1)
        xz = self.conv_pipeout_xz(xz)
            
        return xz.view(-1, 1).squeeze(dim=1)

class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipein = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            )
        
        self.conv_pipelist = nn.ModuleList(
                [nn.Sequential(
                nn.Conv2d(in_channels=2**i*DISCR_FILTERS, out_channels=2**(i+1)*DISCR_FILTERS,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(2**(i+1)*DISCR_FILTERS),
                nn.ReLU()
                ) for i in range(CONV_STACK_NUM)]
            )
        
        self.conv_pipeout = nn.Sequential(
            nn.Conv2d(in_channels=2**CONV_STACK_NUM*DISCR_FILTERS, out_channels=LATENT_VECTOR_SIZE,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        conv_out = self.conv_pipein(x)
        for conv_pipe in self.conv_pipelist:
            conv_out = conv_pipe(conv_out)
        conv_out = self.conv_pipeout(conv_out)
            
        return conv_out

class Decoder(nn.Module):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        
        # pipe deconvolves input vector into (3, IMAGE_SIZE, IMAGE_SIZE) image
        self.conv_pipein = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=2**CONV_STACK_NUM*GENER_FILTERS,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d (2**CONV_STACK_NUM*GENER_FILTERS),
            nn.ReLU()
            )
        
        self.conv_pipelist = nn.ModuleList(
                [nn.Sequential(
                nn.ConvTranspose2d(in_channels=2**(i+1)*GENER_FILTERS, out_channels=2**i*GENER_FILTERS,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d (2**i*GENER_FILTERS),
                nn.ReLU()
                ) for i in range(CONV_STACK_NUM).__reversed__()]
            )
        
        self.conv_pipeout = nn.Sequential(
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            )

    def forward(self, x):
        conv_out = self.conv_pipein(x)
        for conv_pipe in self.conv_pipelist:
            conv_out = conv_pipe(conv_out)
        conv_out = self.conv_pipeout(conv_out)
        return conv_out


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=True, action='store_true',
        help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
    ]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_encoder = Encoder(input_shape=input_shape).to(device)
    net_decoder = Decoder(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam (
        params=list(net_encoder.parameters()) + list(net_decoder.parameters()), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam (
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    writer = SummaryWriter()

    enc_losses = []
    dec_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_decoder(gen_input_v)
        
        output_v = net_encoder(batch_v)
        

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v, output_v.detach())
        dis_output_fake_v = net_discr(gen_output_v.detach(), gen_input_v)
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train decoder and encoder
        gen_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v, output_v)
        dis_output_fake_v = net_discr(gen_output_v, gen_input_v)
        dec_loss_v = objective(dis_output_fake_v, true_labels_v)
        enc_loss_v = objective(dis_output_true_v, fake_labels_v)
        gen_loss_v = dec_loss_v + enc_loss_v
        gen_loss_v.backward()
        gen_optimizer.step()
        dec_losses.append(dec_loss_v.item())
        enc_losses.append(enc_loss_v.item())
        
        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: enc_loss=%.3e, dec_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(enc_losses), np.mean(dec_losses), np.mean(dis_losses))
            writer.add_scalar(
                "enc_loss", np.mean(enc_losses), iter_no)
            writer.add_scalar(
                "dec_loss", np.mean(dec_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(dis_losses), iter_no)
            enc_losses = []
            dec_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64],nrow = 8, normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64],nrow = 8, normalize=True), iter_no)
        if iter_no == 100000:
            break