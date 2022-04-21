import os
import time
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn

from utils import *
from builders.model_builder import build_model
from builders.dataloader_builder import build_dataloaders
from loss import L1Loss2d, CrossEntropyLoss2d, FocalLoss2d, ConsistencyLoss2d, SmoothEdgeLoss
from metrics.iou import IoU
from pytorch_lightning.metrics import MeanAbsoluteError, MeanSquaredError, Accuracy
from scheduler import WarmupPolyLR

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
print(torch_ver)

GLOBAL_SEED = 1234
DEVICE = None

def parse_args():
    parser = ArgumentParser(description='XTC-Semantic Segmentation')
    
    """ model and dataset """
    parser.add_argument('--model', type=str.lower, default="unet", choices=['unet'], help="model name: (default UNet)")
    parser.add_argument('--encoder', type=str.lower, default="resnet34", choices=['unet', 'resnet34'], help="type of decoder")
    parser.add_argument('--downsample', type=int, default=5, choices=[3, 4, 5, 6], help="the number of UNet downsample blocks")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size of model (format: H,W) (default is for UNet)")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--classes', type=int, default=5, help="the number of classes in the dataset")
    parser.add_argument('--valid_size', type=float, default=0.2, help="portion of the training set used for the validation set")
    parser.add_argument('--train_type', type=str, default="baseline", choices=['baseline', 'xtc', 'semi+xtc'], help="how to train the network")
    parser.add_argument('--xtasks', action='store', type=str, nargs='*', default=['depth', 'normal', 'edge'], help="list of cross tasks. Examples: --xtasks item1 item2")
    parser.add_argument('--augment', type=bool, default=True, help="data augmentation")
    parser.add_argument('--in_domain', type=str.lower, default="rgb", choices=['rgb', 'seg'], help="input domain")
    parser.add_argument('--out_domain', type=str.lower, default="seg", choices=['seg', 'depth', 'normal'], help="input domain")
    parser.add_argument('--dataset', type=str.lower, default="ade20k", choices=['ade20k', 'nyuv2', 'ade_nyu', 'dd2', 'door', 'doort'], help="dataset name")
    parser.add_argument('--crf', type=bool, default=False, help="CRF as RNN")
    ### weight initialization
    
    """ training hyper params """
    parser.add_argument('--max_epochs', type=int, default=800, help="the max number of epochs")
    # parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    # parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.5 to 2")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="the batch size")
    parser.add_argument('--optim',type=str.lower, default='adam', choices=['sgd','adam'], help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', choices=['poly', 'warmpoly'], help='name of lr schedule')
    parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0/3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--val_rate', type=int, default=50, help='rate (epochs) of validation')
    
    """ cuda setting """
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")

    """ checkpoint and log """
    parser.add_argument('--resume', type=str, default="", help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--checkpoint', default=20, help="rate of saving model")
    # parser.add_argument('--log_file', default="log.txt", help="storing the training and validation logs")
    
    args = parser.parse_args()

    return args

def train_model(args):
    """
    args:
       args: global arguments
    """
    
    print("=====> Configuration")
    cuda = args.cuda and torch.cuda.is_available()
    DEVICE = get_device(cuda)

    print("Input domain: {}".format(args.in_domain))
    print("Out domain: {}".format(args.out_domain))
    print("Train type: {}".format(args.train_type))

    xmodels = {}

    if not args.train_type == 'baseline':
      print("Consistency tasks: {}".format(args.xtasks))
      
      for task in args.xtasks:
        xmodel = build_model("unet", encoder=args.encoder, in_domain="seg", out_domain=task, num_classes=args.classes, downsample=6)
        checkpoint = torch.load('./models/{}.pth'.format(task))
        xmodel.load_state_dict(checkpoint['model'])
        xmodel.eval()
        xmodels[task] = xmodel


    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("Input size: {}".format(input_size))

    # set the seed
    set_seed(GLOBAL_SEED)
    # print("=====> Setting Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("\n=====> Building network")

    # build the model and initialization
    model = build_model(args.model, encoder=args.encoder, in_domain=args.in_domain, out_domain=args.out_domain, num_classes=args.classes, downsample=args.downsample, crf=args.crf, device=DEVICE)

    if args.model == 'unet' and args.encoder == 'unet':
      initializer = nn.init.kaiming_normal_
      normalizer = nn.GroupNorm

      init_weight(model, initializer, normalizer, bn_eps=1e-3, bn_momentum=0.1, mode='fan_in')

    print("\n=====> Computing network parameters")
    total_paramters = netParams(model)
    print("Number of parameters: %d " % (total_paramters))

    # load data and data augmentation
    print("\n=====> Loading dataset")
    trainLoader, valLoader = build_dataloaders(dataset=args.dataset,
                                               in_domain=args.in_domain,
                                               out_domain=args.out_domain,
                                               input_size=input_size,
                                               batch_size=args.batch_size,
                                               augment=args.augment,
                                               valid_size=args.valid_size,
                                               shuffle=True,
                                               show_sample=False,
                                               num_workers=args.num_workers,
                                               pin_memory=cuda)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter       

    # print('=====> Dataset statistics')
    # print("data['classWeights']: ", datas['classWeights'])
    # print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively
    # weight = torch.from_numpy(datas['classWeights'])

    if args.dataset == 'ade20k':
      hits = [841593329, 453179677, 33077022, 142112364, 182985121]
    elif args.dataset == 'nyuv2':
      hits = [293807951, 95238428, 9624999, 6116738, 40344684]
    elif args.dataset == 'ade_nyu':
      ade_hits = [841593329, 453179677, 33077022, 142112364, 182985121]
      nyu_hits = [293807951, 95238428, 9624999, 6116738, 40344684]
      hits = [a + b for a, b in zip(ade_hits, nyu_hits)]

    # if args.dataset == 'dd2':
    #   weight = [0., 0., 1., 0., 0.]
    # else:
    #   sum_hits = sum(hits)
    #   # weight = [sum_hits/h for h in hits]
    #   weight = [1 - h/sum_hits for h in hits]
    
    # weight = torch.tensor(weight)

    # if cuda:
    #   weight = weight.cuda()

    weight = None
      
    # if args.out_domain == 'seg':
    #   eps = 0.1
    # else:
    #   eps = 0

    print("\n=====> Setting loss function")
    if args.train_type == 'xtc':
      criteria = ConsistencyLoss2d(weight=weight, xtasks=args.xtasks)
      val_criteria = CrossEntropyLoss2d(weight=weight)
    elif args.out_domain in ['seg']:
      #criteria = CrossEntropyLoss2d(weight=weight)
      # criteria = FocalLoss2d(weight=weight)
      criteria = SmoothEdgeLoss(weight=weight)
      #val_criteria = CrossEntropyLoss2d(weight=weight)
      val_criteria = SmoothEdgeLoss(weight=weight)
    elif args.out_domain in ['depth', 'normal']:
      criteria = L1Loss2d()
      val_criteria = L1Loss2d()

    if cuda:
        criteria = criteria.cuda()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            args.gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            args.gpu_nums = 1
            print("Using single GPU for training")
            model = model.cuda()  # 1-card data parallel

        if not args.train_type == 'baseline':
          for xmodel in xmodels.values():
            xmodel.cuda()

    print("Loss: {}".format(criteria.__class__.__name__))

    args.savedir = os.path.join(CHECKPOINT_DIR, args.in_domain + '_' + args.out_domain + '_' + args.model + '_' + args.encoder + '_' + args.train_type + '_' + str(args.batch_size) + '_' + str(args.augment)) + '_' + str(args.crf) + '_' + args.dataset

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0

    if args.out_domain == 'seg':
      best_mIoU = 0

    # continue training
    if args.resume:
        print("\n=====> Loading checkpoint")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            
            print("Loaded checkpoint '{}' (epoch {}) ".format(args.resume, checkpoint['epoch']))

            if args.out_domain == 'seg':
              best_mIoU = checkpoint['best_mIoU']
              print("Best IoU = {}".format(best_mIoU))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True
    # cudnn.deterministic = True ## my add

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    if args.out_domain in ['seg']:
      metric = IoU(args.classes)
      metric2 = Accuracy()
    elif args.out_domain in ['depth', 'normal']:
      metric = MeanAbsoluteError()
      metric2 = None

    args.log_file = args.in_domain + '_' + args.out_domain + '_' + args.model + '_' + args.encoder + '_' + args.train_type + '_' + str(args.batch_size) + '_' + str(args.augment) + '_' + str(args.crf) + '_' + args.dataset + '.txt'
    logFileLoc = os.path.join(LOG_DIR, args.log_file)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Model: %s\nEncoder: %s\nParameters: %s\nInput domain: %s\nOutput domain: %s\nTrain type: %s\nBatch size: %s\nLoss: %s\nSeed: %s" % (model.__class__.__name__, args.encoder,
                                                                                                                                            str(total_paramters),
                                                                                                                                            args.in_domain,
                                                                                                                                            args.out_domain,
                                                                                                                                            args.train_type,
                                                                                                                                            args.batch_size,
                                                                                                                                            criteria.__class__.__name__, 
                                                                                                                                            GLOBAL_SEED))
        if not args.train_type == 'baseline':
          logger.write('Consistency tasks: %s\n' % (str(args.xtasks)))

        if args.out_domain == 'seg':
          logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Learning Rate', 'Loss (Tr)', 'Loss (Val)',  'Mean IoU (Val)', 'Others', 'Wall', 'Door', 'Ceiling', 'Floor', 'Average accuracy'))
        else:                                                                                                                                 
          logger.write("\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'lr', 'Loss (Tr)', 'Loss (Val)', metric.__class__.__name__ + ' (Val)'))
    logger.flush()

    mLoss_tr_list = []
    epoches = []
    val_epoches = []
    metric_val_list = []
    mLoss_val_list = []

    print('\n=====> Beginning training')
    
    total_batches = len(trainLoader)
    print("Number of iterations per epoch:", total_batches)
    
    for epoch in range(start_epoch, args.max_epochs):
        # training

        mLoss_tr, lr = train(args, trainLoader, model, criteria, optimizer, epoch, cuda, xmodels)
        mLoss_tr_list.append(mLoss_tr)

        epoches.append(epoch)

        # validation
        if epoch % args.val_rate == 0 or epoch == (args.max_epochs - 1):
            print("\n=====> Validating")
            if metric.__class__.__name__ == 'IoU':
              metric.reset()
            
            val_epoches.append(epoch)
            mLoss_val, metric_val, class_metric, metric2_val = val(args, valLoader, model, val_criteria, metric, metric2, cuda)
            metric_val_list.append(metric_val)
            mLoss_val_list.append(mLoss_val)
            # record train information
            if args.out_domain == 'seg':
              logger.write("\n%d\t%.7f\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (epoch, lr, mLoss_tr, mLoss_val, metric_val, class_metric[0], 0.0, class_metric[1], 0.0, 0.0, metric2_val))
            else: 
              logger.write("\n%d\t%.7f\t%.4f\t\t%.4f\t\t%.4f" % (epoch, lr, mLoss_tr, mLoss_val, metric_val))
            logger.flush()
            print("Epoch No.: %d\tTrain Loss = %.5f\t Val Loss = %.5f\t %s(val) = %.4f\t Learning Rate = %.6f\n" % (epoch + 1,
                                                                                        mLoss_tr, mLoss_val, metric.__class__.__name__,
                                                                                        metric_val, lr))
        else:
            # record train information
            logger.write("\n%d\t\t%.7f\t\t%.4f" % (epoch, lr, mLoss_tr))
            logger.flush()
            print("Epoch No.: %d\tTrain Loss = %.5f\t Learning Rate = %.6f\n" % (epoch + 1, mLoss_tr, lr))

        if epoch % args.checkpoint == 0 or epoch == (args.max_epochs - 1):
            # save the model
            model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
            state = {"epoch": epoch + 1, "best_mIoU": best_mIoU, "model": model.state_dict()}

            torch.save(state, model_file_name)

        model_file_name = args.savedir + '/model_final.pth'
        state = {"epoch": epoch + 1, "best_mIoU": best_mIoU, "model": model.state_dict()}

        torch.save(state, model_file_name)

        if args.out_domain == 'seg' and metric_val > best_mIoU:
          print("Saving new best model: {} -> {}".format(best_mIoU, metric_val))
          best_mIoU = metric_val
          model_file_name = args.savedir + '/model_best.pth'
          state = {"epoch": epoch + 1, "best_mIoU": best_mIoU, "model": model.state_dict()}

          torch.save(state, model_file_name)


        # draw plots for visualization
        if epoch % args.val_rate == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per 50 epochs
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(epoches, mLoss_tr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")

            plt.savefig(args.savedir + " (loss_vs_epochs).png")

            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(val_epoches, metric_val_list, label=metric.__class__.__name__)
            ax2.set_title("Metric vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Metric")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + " (val_metric_vs_epochs).png")

            plt.clf()

            fig3, ax3 = plt.subplots(figsize=(11, 8))

            ax3.plot(val_epoches, mLoss_val_list)
            ax3.set_title("Average validation loss vs epochs")
            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Loss")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + " (val_loss_vs_epochs).png")

            plt.close('all')

    logger.close()

def train(args, train_loader, model, criterion, optimizer, epoch, cuda, xmodels):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    total_batches = len(train_loader)
    epoch_loss = []

    print('-> epoch[%d/%d]' % (epoch + 1, args.max_epochs))

    st = time.time()
    for iteration, batch in enumerate(train_loader, 0):

        args.cur_iter = epoch * args.per_iter + iteration
        # learming scheduling
        if args.lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.9)

        lr = optimizer.param_groups[0]['lr']

        start_time = time.time()
        x = batch['x']
        y = batch['y']

        if cuda:
            x = x.cuda()
            y = y.cuda()

        y_ = model(x)

        if not args.train_type == 'baseline':
          xoutputs = {}
          xtargets = {}

          seg = torch.unsqueeze(y, 1).float()

          seg_ = torch.argmax(y_, axis=1)
          seg_ = torch.unsqueeze(seg_, 1).float()

          for task in args.xtasks:
            xmodel = xmodels[task]
            xoutput = xmodel(seg_)
            xtarget = xmodel(seg)
            xoutputs[task] = torch.squeeze(xoutput, 1)
            xtargets[task] = torch.squeeze(xtarget, 1)

          loss = criterion(y_, y, xoutputs, xtargets)
        
        else:

          loss = criterion(y_, y, batch)

        # y = y.squeeze(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time


        print('iter: (%d/%d)     cur_lr: %.6f     loss: %.5f     time: %.2f' % (iteration + 1, total_batches,
                                                                    lr, loss.item(), time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("\nRemaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr

def val(args, val_loader, model, criterion, metric, metric2, cuda):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    metrics = []
    losses = []

    for i, batch in enumerate(val_loader):
        start_time = time.time()

        x = batch['x']
        y = batch['y']

        with torch.no_grad():
            # input_var = Variable(input).cuda()

            if cuda:
              x = x.cuda()
              y = y.cuda()
            
            # label = label.squeeze(1)
            y_ = model(x)
            loss = criterion(y_, y, batch)
            losses.append(loss.item())

        time_taken = time.time() - start_time
        print("iter: (%d/%d) \ttime: %.2f" % (i + 1, total_batches, time_taken))
        # output = output.cpu().data[0].numpy()
        # gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        # output = output.transpose(1, 2, 0)
        # output = torch.argmax(output, dim=1)

        if args.out_domain == 'seg':
          metric.add(y_.detach(), y.detach())
          y_ = torch.argmax(y_, 1)
          batch_acc = metric2(y_.cpu().detach(), y.cpu().detach())
        else:
          y_ = y_.squeeze(1)
          val_metric = metric(y_.cpu().detach(), y.cpu().detach())
          metrics.append(val_metric)
        # iou, class_iou = get_iou(output, label, args.classes)
        # ious.append(iou)

    if args.out_domain == 'seg':
      class_IoU, mMetric = metric.value()
      print(class_IoU)
      mMetric2 = metric2.compute()
    else:
      mMetric = sum(metrics) / len(metrics)
      mMetric2 = None
      class_IoU = None

    mLoss = sum(losses) / len(losses)

    return mLoss, mMetric, class_IoU, mMetric2

if __name__ == '__main__':
    
    start = time.time()
    args = parse_args()

    train_model(args)
    
    end = time.time()

    elapsed = end - start
    hour = elapsed // 3600
    minute = (elapsed % 3600) // 60
    second = int(elapsed - hour * 3600 - minute * 60)
    print("Training time: {} hours {} minutes {} seconds".format(hour, minute, second))
