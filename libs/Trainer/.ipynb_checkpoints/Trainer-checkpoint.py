import time
import torch
from torch.cuda import amp
import sys
sys.path.append("../utils")
from eval import AverageMeter
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, evalate, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = amp.GradScaler() 
        self.device = device
        self.config = config
        self.evalate = evalate
        self.t_losses = AverageMeter()
        self.v_losses = AverageMeter()
        self.writer = SummaryWriter(log_dir="./logs/"+config.f_name)
        self.iter = 0
        
    def fit(self, train_dl, val_dl):
        best_eval = 0.0
        for i in range(self.config.n_epochs):
            start = time.time()
            self.train_one_epoch(train_dl)
            self.validation(val_dl)
            #lr = self.optimizer.param_groups[0]['lr']
            metric = self.evalate.get_scores()[0]['mIoU']
            self.log(f'[RESULT]: Epoch: {i+1}, train_loss: {self.t_losses.avg:.4f}, val_loss: {self.v_losses.avg:.5f}, mIoU: {metric:.6f}, time: {(time.time() - start):.3f}')
            self.writer.add_scalar('train_loss', round(self.t_losses.avg, 5), i+1)
            self.writer.add_scalar('val_loss', round(self.v_losses.avg, 5), i+1)
            self.writer.add_scalar('mIoU', round(metric, 6), i+1)
            
            if best_eval < metric:
                best_eval = metric
                self.save(epoch=i+1, mIoU=best_eval)
        self.save(epoch=self.config.n_epochs, mIoU=metric, last=True)
        self.writer.close()
        
        
    def train_one_epoch(self, train_dl):
        self.model.train()
        self.t_losses.reset()
        for img, target in train_dl:
            self.optimizer.zero_grad()
            with amp.autocast(enabled=True):
                pred = self.model(img.to(device))
                loss = self.criterion(pred, target.to(device).long())
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.t_losses.update(loss.item(), self.config.train_batch_size)
            
            if (self.iter % 10) == 0 :
                print(f'iter : {self.iter}')
            self.iter += 1
            
            
    def validation(self, val_dl):
        self.model.eval()
        self.evalate.reset()
        self.v_losses.reset()
        for img, target in val_dl:
            with amp.autocast(enabled=True):
                pred = self.model(img.to(device))
                loss = self.criterion(pred, target.to(device).long())
            pred = pred['out'].cpu()
            pred = torch.argmax(pred.squeeze(0), dim=1, keepdim=True).squeeze(1).numpy()
            target = target.cpu().numpy()
            self.evalate.update(pred=pred, gt=target)
            self.v_losses.update(loss.item(), self.config.val_batch_size)
            
    def save(self, epoch, mIoU, last=False):
        if last:
            l_or_b = '_last.bin'
        else:
            l_or_b = '_best.bin'
        torch.save({
            'model_state_dict': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'mIoU': mIoU,
        }, config.weight_path + l_or_b)
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(config.log_path+'.txt', mode='a') as logger:
            logger.write(f'{message}\n')
            
if __name__ == '__main__':
    from eval import runningScore
    import torch
    import torch.nn as nn
    import torch.utils.data as data
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import _LRScheduler
    
    sys.path.append("../../datasets")
    sys.path.append("../transforms")
    sys.path.append("../models")
    import ext_transforms as et
    from Cityscapes_dataset import Cityscapes 
    from DDRNet_23_slim import get_seg_model
    from loss import AuxLoss
    
    
    class DDRmodel(nn.Module):
        def __init__(self, model):
            super(DDRmodel, self).__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            out_width = x.shape[-1] 
            out_height = x.shape[-2]

            out['out'] = F.interpolate(out['out'], size=[out_height, out_width], mode='bilinear')
            out['aux'] = F.interpolate(out['aux'], size=[out_height, out_width], mode='bilinear')
            return out
        
    class TrainGlobalConfig:
        num_classes: int = 19
        n_epochs: int = 30
        lr: float = 0.01
        crop_size = (64, 64)
        train_batch_size: int = 8
        val_batch_size: int = 2
        
        w_folder = 'weights/'
        l_folder = 'logs/'
        f_name = 'ddrnet_test'

        weight_path = w_folder + f_name 
        log_path = l_folder + f_name 
        
        verbose = True
        verbose_step = 1
        
    class PolyLR(_LRScheduler):
        def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
            self.power = power
            self.max_iters = max_iters  # avoid zero lr
            self.min_lr = min_lr
            super(PolyLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                    for base_lr in self.base_lrs]
        
    config = TrainGlobalConfig
    train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=config.crop_size, padding=255, pad_if_needed=True),
            #et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_transform = et.ExtCompose([
        et.ExtRandomCrop(size=config.crop_size, padding=255, pad_if_needed=True),
        et.ExtToTensor(),
        et.ExtNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    root = '/home/sugimoto/datasets/cityscapes'
    train_dataset = Cityscapes(root=root, split='train', mode='fine', target_type='semantic', transforms=train_transform)
    val_dataset = Cityscapes(root=root, split='val', mode='fine', target_type='semantic', transforms=val_transform)
    train_dl = data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dl = data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
    
    evalate = runningScore(n_classes=config.num_classes)
    device = torch.device('cuda:0')
    
    model = get_seg_model(1, augment=True)
    model = DDRmodel(model)
    criterion = AuxLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = PolyLR(optimizer, config.n_epochs*len(train_dl), power=0.9)
    a = Trainer(model, optimizer, scheduler, criterion, evalate, device, config)
    a.fit(train_dl, val_dl)
 