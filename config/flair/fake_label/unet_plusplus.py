from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.flair_dataset import *
from geoseg.models.Unet_plusplus import UnetPlusPlus
# from catalyst.contrib.optimizers import Lookahead
from catalyst.contrib.nn import Lookahead
from catalyst import utils

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# training hparam
max_epoch = 3
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "UNet-V1-FakeLabel"
weights_path = "model_weights/flair/{}".format(weights_name)
test_weights_name = "UNet-V1-FakeLabel"
log_name = 'flair/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [2]
strategy = None
pretrained_ckpt_path = 'model_weights/flair/UNet-UPLoss-Epoch10/UNet-UPLoss-Epoch10.ckpt'
resume_ckpt_path = None
#  define the network
net = UnetPlusPlus(num_classes=num_classes)

# define the loss
loss = UnetPlusPlusLoss(ignore_index=ignore_index)
use_aux_loss = True


# define the fake label dataloader
train_dataset = FLAIRFakeDataset(mode='train')
val_dataset = FLAIRFakeDataset(mode='val')
test_dataset = FLAIRTestDataset(mode='test')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)