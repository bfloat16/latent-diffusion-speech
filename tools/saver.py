import os
import time
import yaml
import datetime
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from . import utils
from torch.utils.tensorboard import SummaryWriter

class Saver(object):
    def __init__(self, args, initial_global_step=-1):

        self.expdir = args['diffusion']['train']['expdir']
        self.sample_rate = args['data']['sampling_rate']
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        os.makedirs(self.expdir, exist_ok=True)

        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        os.makedirs(self.expdir, exist_ok=True)

        self.writer = SummaryWriter(self.expdir)

        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)

    def log_info(self, msg):
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg

        print(msg_str)

        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str + '\n')

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)

    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        spec = spec_cat[0]
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 9))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        self.writer.add_figure(name, fig, self.global_step)

    def log_audio(self, dict):
        for k, v in dict.items():
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)

    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_model(
            self,
            model,
            optimizer,
            name='model',
            postfix=''):
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.expdir, name + postfix + '.pt')

        if type(model) is torch.nn.parallel.distributed.DistributedDataParallel or type(model) is torch.nn.parallel.DistributedDataParallel:
            model = model.module
            
        # save
        if optimizer is not None:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path_pt)
        else:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict()}, path_pt)

    def delete_model(self, name='model', postfix=''):
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.expdir, name + postfix + '.pt')

        if os.path.exists(path_pt):
            os.remove(path_pt)

    def global_step_increment(self):
        self.global_step += 1

class Saver_empty(object):
    def __init__(self,args,initial_global_step=-1):

        pass

    def log_info(self, msg):
        pass

    def log_value(self, dict):
        pass

    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):
        pass

    def log_audio(self, dict):
        pass

    def get_interval_time(self, update=True):
        pass

    def get_total_time(self, to_str=True):
        pass

    def save_model(
            self,
            model,
            optimizer,
            name='model',
            postfix='',
            to_json=False):
        pass

    def delete_model(self, name='model', postfix=''):
        pass

    def global_step_increment(self):
        pass