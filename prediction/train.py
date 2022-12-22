from datetime import datetime
import argparse
import torch,os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from torch.utils.tensorboard import SummaryWriter

from datasets import RacingDataset

# tensorboard --logdir F:\repositories\misc\tensorboard-logs\prediction_racing\
# python .\prediction\train.py --n_epochs 100


parser = argparse.ArgumentParser()
parser.add_argument("--num_class", type=int, default=18, help="number of epochs of training")
parser.add_argument("--input_size", type=int, default=55, help="number of epochs of training")
parser.add_argument("--pretrained", action='store_true', help="Determine whether to use a pre-trained model")

parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--batch_size_val", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--logs_dir", type=str, default='F:/repositories/misc/tensorboard-logs/prediction_racing', help="logs dir")

opt = parser.parse_args()
print(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def main(): 

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(opt.input_size),
        transforms.Normalize(mean=0.5, std=0.2)])

    train_datasets = RacingDataset(os.path.join('data/csv/', 'train.csv'), transform=t)
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=opt.batch_size, shuffle=True)
    
    val_datasets = RacingDataset(os.path.join('data/csv/', 'val.csv'), transform=t)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=opt.batch_size_val, shuffle=False)

    save_tag = '-resnet50-pre-false-inited-e100-time2'

    train_data_len = train_datasets.__len__()
    val_total = val_datasets.__len__()
    steps = train_data_len // opt.batch_size + 1
    logs_dir = os.path.join(opt.logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + save_tag)
    # if opt.resume:
    # init model
    # model = models.resnet18(pretrained=opt.pretrained)
    model = models.resnet50(pretrained=opt.pretrained)
    # model = models.resnet101(pretrained=opt.pretrained)
    if opt.pretrained:
        # 除了bn层，将所有的参数层进行冻结
        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False
    else:
        init_weights(model.modules())

    # 输入channel,width,height不同, 需要重新定义conv1
    model.conv1 = nn.Conv2d(opt.num_class, 64, kernel_size=1, stride=1)

    # 定义一个新的FC层
    num_fc_ftr = model.fc.in_features
    model.fc = nn.Linear(num_fc_ftr, opt.num_class)

    # else:
    #    model = torch.load('F:/repositories/workspaces/prediction_racing/prediction/checkpoint/ckp_epoch(1)_2021_07_25_1226.pth', map_location=device)
    model = model.to(device)

    # ----------
    #  Training
    # ----------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.n_epochs):

        model.train()
        for i, (data, target, odds) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)
            target = torch.where(target < 1, 0., 1.)

            # Forward pass
            target_hat = model(data)
            loss = criterion(target_hat, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0:
                print('Epoch [{:>3}/{:>3}]  Batch [{:>3}/{:>3}]  Loss: {:.10f}'.format(epoch + 1, opt.n_epochs, i + 1, steps, loss.item()))
                draw_chart('01_loss', loss.item(), steps * epoch + i + 1, logs_dir)

            # if i > 10:
            #    break
            #if (epoch + 1) % 5 == 0: 
            # based on 100 point
        point_base, accuracy, val_loss, point_expense, point_income, blance = 100, 0, 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            
            for v_i, (v_data, v_target, v_odds) in enumerate(val_loader):          

                v_data = v_data.to(device)
                v_target = v_target.to(device)
                v_odds = v_odds.to(device)
                
                # optimizer.zero_grad()
                v_target_hat = model(v_data)
                 # sum up batch loss
                val_loss += criterion(v_target_hat, v_target).item()

                # set top 1 position to compute point
                top_target_hat = torch.topk(v_target_hat, 1)[1]
                top_target_hat = torch.squeeze(F.one_hot(top_target_hat, num_classes=18))

                # set the data of the top 3 to 1, the prediction result is in top 3 to be successed.
                ret = torch.where(v_target < 1, 0, 1) * top_target_hat

                accuracy += torch.count_nonzero(ret).item()
                # expense
                point_expense += torch.count_nonzero(top_target_hat).item() * point_base
                # convert to multiple odds 
                v_mult_odds = torch.where(v_odds == 0, 0., v_odds * 0.12 + 1)
                # income
                point_income += int(torch.sum(ret * v_mult_odds * point_base).item())

                blance = point_income - point_expense

        acc_rate = round(100. * (accuracy / val_total), 1)
        refund_rate = round(100. * (point_income / point_expense), 1)
        print(' +++++++++++++++++++++++++++++++++++ ')
        print(' +++++  Accuracy  {:>6}/{}  +++++'.format(accuracy, val_total))
        print(' +++++  Acc Rate       {:>5}%  +++++'.format(str(acc_rate)))
        print(' +++++  Val loss      {:.5f}  +++++'.format((val_loss / steps)))
        print(' +++++  Expense    {:>10}  +++++'.format(point_expense))
        print(' +++++  Income     {:>10}  +++++'.format(point_income))
        print(' +++++  Blance     {:>10}  +++++'.format(blance))
        print(' +++++  Refund rate    {:>5}%  +++++'.format(str(refund_rate)))
        print(' +++++++++++++++++++++++++++++++++++ ')
        
        draw_chart('02_acc_rate(%)', acc_rate, epoch + 1, logs_dir)
        draw_chart('03_refund_rate(%)', refund_rate, epoch + 1, logs_dir)

    checkpoint_filename = "ckp_{}-{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H%M"), save_tag)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    torch.save(model, os.path.join(dir_path, 'checkpoint', checkpoint_filename))
    print(" checkpoint file {} saved.".format(checkpoint_filename))

def draw_chart(name, value, steps, logs_dir):

    with SummaryWriter(log_dir=logs_dir, flush_secs=2, comment='train') as writer:
        
        # writer.add_histogram('his/loss', loss_value, epoch)
        writer.add_scalar(name, value, steps)

        #writer.add_histogram('his/y', y, epoch)
        #writer.add_scalar('data/y', y, epoch)
        #writer.add_scalar('data/loss', loss, epoch)
        #writer.add_scalars('data/data_group', {'x': x, 'y': y}, epoch)

if __name__ == '__main__':
    main()
