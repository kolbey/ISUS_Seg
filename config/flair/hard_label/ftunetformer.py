from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.flair_hard_dataset import *
from geoseg.datasets.flair_dataset import FLAIRTestDataset
from geoseg.models.FTUNetFormer import ft_unetformer
# from catalyst.contrib.optimizers import Lookahead
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 10
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "FT-V1-HardLabel"
weights_path = "model_weights/flair/{}".format(weights_name)
test_weights_name = "FT-V1-HardLabel"
log_name = 'flair/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [3]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader
train_dataset = FLAIRHardDataset(mode='train')

val_dataset = FLAIRHardDataset(mode='val')

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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)