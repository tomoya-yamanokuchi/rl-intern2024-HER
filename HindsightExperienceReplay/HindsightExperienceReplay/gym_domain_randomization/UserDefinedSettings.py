import os
import warnings
import datetime
import torch
import numpy as np
import random

warnings.simplefilter('ignore', FutureWarning)  # noqa


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True


class UserDefinedSettings(object):

    def __init__(self):
        gpu_num = 0
        self.DEVICE = torch.device(
            "cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu")
        # HalfCheetah, Pendulum, SwingUp, HalfCheetah, Excavator_v1, Excavator_v2, Excavator_v3
        self.ENVIRONMENT_NAME = 'Pendulum'
        self.REWARD_FUNCTION_NAME = 'v99'  # v1, v2, v3, v4, v5
        current_time = datetime.datetime.now()
        file_name = 'M{:0=2}D{:0=2}H{:0=2}M{:0=2}'.format(
            current_time.month, current_time.day, current_time.hour, current_time.minute)
        self.LOG_DIRECTORY = os.path.join(
            'logs', self.ENVIRONMENT_NAME, 'sac', file_name)
        self.LSTM_FLAG = False
        self.DOMAIN_RANDOMIZATION_FLAG = False

        self.num_steps = 1e6
        self.batch_size = 16  # 64
        self.lr = 1e-4
        self.learning_rate = self.lr
        self.HIDDEN_NUM = 128
        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_tuning_scale = 1  # 1
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.updates_per_step = 1
        self.policy_update_start_steps = 150 * 20  # 300
        self.target_update_interval = 1  # episode num
        self.evaluate_interval = 10  # episode num
        self.initializer = 'xavier'
        self.run_num_per_evaluate = 5
        self.average_num_for_model_save = 10
        self.learning_episode_num = 400  # 300
        self.learning_episode_num_all_domain = 1000
        self.LEARNING_REWARD_SCALE = 1.

        # distillation parameters
        self.DOMAIN_DIVIDED_NUM = 8  # 8
        self.learning_episode_num_per_each_domain = self.learning_episode_num
        self.distillation_epoch_num = 500
        self.distillation_batch_size = 32

        self.ACTION_DISCRETE_FLAG = False
