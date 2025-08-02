import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):         # 应该和iter类似  变成可迭代 一个个取数据
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']    # 2
        self.save_cycle = config['solver']['save_cycle']                        # 1200
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        
        '这个ema里有模型 后面会用于采样样本 self.ema.ema_model.generate_mts '
        '会用到模型 Models.......Diffusion_TS 的generate_mts函数'
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
                        
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt    # 配置里添加了一个优化器参数
        self.sch = instantiate_from_config(sc_cfg)   # 导入了 engine.lr_sch.ReduceLROnPlateauWithWarmup
                                       # 改进的Adam 有自适应学习率功能

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        
#         if self.logger is not None:
#             tic = time.time()
#             self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

            
        with tqdm(initial=step, total=self.train_num_steps) as pbar:    # 这里的应该是iteration 不是epoch
            
            while step < self.train_num_steps:
                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):    # 2  循环2次 然后损失要分半 
                    
                    data = next(self.dl).to(device)           # [128, 24, 5]  前面对dl用了cycle 类似iter 可以取数据
                    
                    loss = self.model(data, target=data)        # 计算损失 可以加上频域损失
                    
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()                      # 梯度反传了 但是还没有用step 模型还没更新 后面要剪裁
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                '剪裁参数 再优化器step  新建的优化器也step'
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                
#                 with torch.no_grad():             # 每1200个iteration记录一次
#                     if self.step != 0 and self.step % self.save_cycle == 0:
#                         self.milestone += 1      # 这个就是个数字
#                         self.save(self.milestone)  # 这里会记录模型参数 优化器 ema
# #                         self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
#                     if self.logger is not None and self.step % self.log_frequency == 0:
#                         info = '{}: train'.format(self.args.name)
#                         info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
#                         info += ' ||'
#                         info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
#                         info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
#                         info += ' | Total Loss: {:.6f}'.format(total_loss)
#                         self.logger.log_info(info)
#                         self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

#                 pbar.update(1)

        print('training complete')
#         if self.logger is not None:
#             self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

            
    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
            
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1       # 分几次生成

        for _ in range(num_cycle):
            
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)  # Diffusion_TS模块里的generate_mts
            
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])   # 拼接
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    
    
    
    
    import time
    
    'DPS采样'
    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
            
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])    # 缺失掩码 

        for idx, (x, t_m) in enumerate(raw_dataloader):  # 取出数据和掩码 掩码用true false表示
            x, t_m = x.to(self.device), t_m.to(self.device)
            
            
            
            # 可选常规ddpm采样或快速ddim采样 根据具体设定的采样步和总的扩散步的大小
            if sampling_steps == self.model.num_timesteps:          # target是观测值
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target = x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                      
                                                               sampling_timesteps=sampling_steps)


            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples
