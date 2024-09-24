import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import SQSFormer as SQSFormer
import utils
from models.SQSFormer import RWKV, RWKV_Init, _weights_init
from utils import recorder
from evaluation import HSIEvaluation
import itertools
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import device
import math, sys, datetime
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)


class BaseTrainer(object):
    # max_epochs = 50
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    weight_decay = 0.01
    lr_decay = False  # linear warmup followed by cosine decay
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper
    final_tokens = 260e9  # at which point do we reach lr_final
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device 
        self.evalator = HSIEvaluation(param=params)
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None
        self.real_init()

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        epochs = self.params['train'].get('epochs')
        print(epochs)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        # 打印 train_loader 中的批次数量
        print(f"Total number of batches in train_loader: {len(train_loader)}")

        # 检查 train_loader 中的一个 batch 数据
        for i, (data, target) in enumerate(train_loader):
            print(f"Batch {i + 1}")
            print(f"Data shape: {data.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Sample labels: {target[:10]}")  # 打印前10个标签
            break  # 只检查第一个 batch，避免打印太多数据

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")  # 输出当前 epoch
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()  #优化器
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 26 == 0:
                print("开始测试")
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs))
            if len(outputs.shape) == 1:
                continue
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test


class SQSFormerTrainer(BaseTrainer):
    def __init__(self, params):
        super(SQSFormerTrainer, self).__init__(params)

    def real_init(self):
        # 先初始化网络 self.net
        self.net = SQSFormer.SQSFormer(self.params).to(self.device)

        RWKV_Init(self.net)  # 确保传入的是 self.net，它是 nn.Module 实例
        # self.net.apply(_weights_init)
        # loss
        self.criterion = nn.CrossEntropyLoss()      #交叉熵损失函数
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits, A_vecs, B_vecs = outputs

        loss_main = nn.CrossEntropyLoss()(logits, target)
        return loss_main



def get_trainer1(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "rwkvformer":
        return SQSFormerTrainer(params)
    assert Exception("Trainer not implemented!")

#
# class TrainerConfig(object):
#     print("TrainerConfig")
#     max_epochs = 20
#     batch_size = 64
#     learning_rate = 4e-4
#     betas = (0.9, 0.99)
#     eps = 1e-8
#     grad_norm_clip = 1.0
#     weight_decay = 0.01
#     lr_decay = False  # linear warmup followed by cosine decay
#     warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper
#     final_tokens = 260e9  # at which point do we reach lr_final
#     epoch_save_frequency = 0
#     epoch_save_path = 'trained-'
#     num_workers = 0  # for DataLoader
#
#
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)



# class Trainer(BaseTrainer):
#     def __init__(self, model, train_dataset, test_dataset, config , params, wandb=None):
#         super(Trainer, self).__init__()
#         self.params = params
#         self.net_params = params['net']
#         self.train_params = params['train']
#         self.device = device
#         self.evalator = HSIEvaluation(param=params)
#         self.net = None
#         self.criterion = None
#         self.optimizer = None
#         self.clip = 15
#         self.unlabel_loader=None
#         self.real_init()
#         self.model = model
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
#         self.config = config
#         self.avg_loss = -1
#         self.steps = 0
#         # if 'wandb' in sys.modules:
#         #     cfg = model.config
#         #     for k in config.__dict__:
#         #         setattr(cfg, k, config.__dict__[k])  # combine cfg
#         #     wandb.init(project="RWKV-LM",
#         #                name=self.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
#         #                config=cfg, save_code=False)
#         #
#         # self.device = 'cpu'
#         # if torch.cuda.is_available():  # take over whatever gpus are on the system
#         #     self.device = torch.cuda.current_device()
#         #     self.model = torch.nn.DataParallel(self.model).to(self.device)
#
#     def real_init(self):
#         # net
#         self.net = SQSFormer.SQSFormer(self.params).to(self.device)    ##修改！
#         # loss
#         self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#         # optimizer
#         self.lr = self.train_params.get('lr', 0.001)
#         self.weight_decay = self.train_params.get('weight_decay', 5e-3)
#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#     def get_run_name(self):
#         raw_model = self.model.module if hasattr(self.model, "module") else self.model
#         cfg = raw_model.config
#         run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + cfg.model_type + '-' + str(
#             cfg.n_layer) + '-' + str(cfg.n_embd)
#         return run_name
#
#     def train1(self):
#         model, config = self.model, self.config
#         raw_model = model.module if hasattr(self.model, "module") else model
#         optimizer = raw_model.configure_optimizers(config)
#
#         def run_epoch(split, wandb=None):
#             is_train = split == 'train'
#             model.train1(is_train)
#             data = self.train_dataset if is_train else self.test_dataset
#             loader = DataLoader(data, shuffle=True, pin_memory=True,
#                                 batch_size=config.batch_size,
#                                 num_workers=config.num_workers)
#
#             pbar = tqdm(enumerate(loader), total=len(loader),
#                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)
#
#             for it, (x, y) in pbar:
#                 x = x.to(self.device)  # place data on the correct device
#                 y = y.to(self.device)
#
#                 with torch.set_grad_enabled(is_train):
#                     _, loss = model(x, y)  # forward the model
#                     loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
#
#                 if is_train:  # backprop and update the parameters
#                     model.zero_grad()
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
#                     optimizer.step()
#
#                     if config.lr_decay:  # decay the learning rate based on our progress
#                         self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
#                         lr_final_factor = config.lr_final / config.learning_rate
#                         if self.tokens < config.warmup_tokens:
#                             # linear warmup
#                             lr_mult = lr_final_factor + (1 - lr_final_factor) * float(self.tokens) / float(
#                                 config.warmup_tokens)
#                             progress = 0
#                         else:
#                             # cosine learning rate decay
#                             progress = float(self.tokens - config.warmup_tokens) / float(
#                                 max(1, config.final_tokens - config.warmup_tokens))
#                             # progress = min(progress * 1.1, 1.0) # more fine-tuning with low LR
#                             lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(
#                                 math.pi * progress)  # better 1.0 ~ 0.1
#                         lr = config.learning_rate * lr_mult
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] = lr
#                     else:
#                         lr = config.learning_rate
#
#                     now_loss = loss.item()  # report progress
#
#                     if 'wandb' in sys.modules:
#                         wandb.log({"loss": now_loss}, step=self.steps * self.config.batch_size)
#                     self.steps += 1
#
#                     if self.avg_loss < 0:
#                         self.avg_loss = now_loss
#                     else:
#                         # factor = max(1.0 / 300, 1.0 / math.sqrt(it + 1))
#                         factor = 1 / (it + 1)
#                         self.avg_loss = self.avg_loss * (1.0 - factor) + now_loss * factor
#                     pbar.set_description(
#                         f"epoch {epoch + 1} progress {progress * 100.0:.2f}% iter {it}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {lr:e}")
#
#         while True:
#             self.tokens = 0  # counter used for learning rate decay
#             for epoch in range(config.max_epochs):
#                 print("run_epoch")
#                 run_epoch('train')
#
#                 if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (
#                         epoch == config.max_epochs - 1):
#                     print(f"最大的maxepoch: {config.max_epochs}")
#                     raw_model = self.model.module if hasattr(self.model,
#                                                              "module") else self.model  # DataParallel wrappers keep raw model object in .module
#                     torch.save(raw_model, self.config.epoch_save_path + str(epoch + 1) + '.pth')
def get_trainer(params):
    trainer_type = params['net']['trainer']
    print("测试位置3")
    print(params)
    if trainer_type == "sqsformer":
        return SQSFormerTrainer(params)
    assert Exception("Trainer not implemented!")


# def get_trainer2(train_dataset, test_dataset, config, params):
#     trainer_type = params['net']['trainer']
#     print(params)
#     if trainer_type == "rwkvformer":
#         return Trainer(train_dataset, test_dataset, config, params)
#     assert Exception("Trainer not implemented!")