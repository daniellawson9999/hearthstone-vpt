import numpy as np
import torch as th

import time

class Trainer:
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.device = device

        self.start_time = time.time()

    
    def train_iteration(self, num_steps, iter_num=0, print_logs=True):
        train_losses = []
        logs = {}

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start

        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)


        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
    
    def train_step(self):
        states, actions, mask = self.get_batch(self.batch_size)

        max_len = states.shape[1]
        first = th.from_numpy(np.zeros((self.batch_size, max_len),dtype=bool)).to(self.device)
        hidden_state = self.model.initial_state(self.batch_size)
        
        # Forward pass
        (translation_actions, click_dists, click_logits, logp_actions), _ = self.model.forward(
            states, first, hidden_state, actions
        )

        # Mask output, compute loss
        mask = mask.reshape(-1)
        target_translation_actions = actions[:,:,:2].reshape(-1, 2)[mask > 0]
        translation_actions = translation_actions.reshape(-1, 2)[mask > 0]
        logp_actions = logp_actions.reshape(-1)[mask > 0]
        loss = self.loss_fn(translation_actions, target_translation_actions, logp_actions)

        # perform backpropation, gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # TODO, could clip gradient
        self.optimizer.step()

        return loss.detach().cpu().item()

