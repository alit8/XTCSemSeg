# import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from crfseg import CRF

from utils import *
from builders.model_builder import build_model
# from builders.dataloader_builder import build_dataloaders
# from loss import L1Loss2d, CrossEntropyLoss2d, FocalLoss2d, ConsistencyLoss2d
# from metrics.iou import IoU
# from pytorch_lightning.metrics import MeanAbsoluteError, MeanSquaredError, Accuracy
# from scheduler import WarmupPolyLR

# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

import glob
import os
import cv2

import torchvision.transforms.functional as TF
from torchvision import transforms

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
print(torch_ver)

GLOBAL_SEED = 1234
DEVICE = None

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation demo')
    
    """ model and dataset """
    parser.add_argument('--model', type=str.lower, default="unet", choices=['unet'], help="model name: (default UNet)")
    parser.add_argument('--encoder', type=str.lower, default="resnet34", choices=['unet', 'resnet34'], help="type of decoder")
    parser.add_argument('--downsample', type=int, default=5, choices=[3, 4, 5, 6], help="the number of UNet downsample blocks")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size of model (format: H,W) (default is for UNet)")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--classes', type=int, default=5, help="the number of classes in the dataset")
    parser.add_argument('--train_type', type=str, default="baseline", choices=['baseline', 'xtc', 'semi+xtc'], help="how to train the network")
    parser.add_argument('--xtasks', action='store', type=str, nargs='*', default=['depth', 'normal', 'edge'], help="list of cross tasks. Examples: --xtasks item1 item2")
    parser.add_argument('--in_domain', type=str.lower, default="rgb", choices=['rgb', 'seg'], help="input domain")
    parser.add_argument('--out_domain', type=str.lower, default="seg", choices=['seg', 'depth', 'normal'], help="input domain")
    parser.add_argument('--crf', type=bool, default=False, help="using CRF as postprocessing")
    """ cuda setting """
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0)")

    """ checkpoint and input """
    parser.add_argument('--checkpoint', type=str, default="", help="use this file to load trained model")
    parser.add_argument('--input', type=str, default="", help="path to video or image or directory of images")
    parser.add_argument('--output', type=str, default="./output", help="path to a directory where output images are saved")

    args = parser.parse_args()

    return args

def get_model(args):
    """
    args:
       args: global arguments
    """
    
    print("=====> Configuration", args.crf)
    cuda = args.cuda and torch.cuda.is_available()
    DEVICE = get_device(cuda)

    print("Input domain: {}".format(args.in_domain))
    print("Out domain: {}".format(args.out_domain))
    print("Model type: {}".format(args.train_type))

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("Input size: {}".format(input_size))

    # set the seed
    set_seed(GLOBAL_SEED)
    # print("=====> Setting Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("\n=====> Building network")

    # build the model and initialization
    model = build_model(args.model, encoder=args.encoder, in_domain=args.in_domain, out_domain=args.out_domain, num_classes=args.classes, downsample=args.downsample, crf=False, device=DEVICE)

    print("\n=====> Computing network parameters")
    total_paramters = netParams(model)
    print("Number of parameters: %d " % (total_paramters))    

    # continue training
    if args.checkpoint:
        print("\n=====> Loading checkpoint")
        assert os.path.isfile(args.checkpoint), "No checkpoint found at '{}'".format(args.checkpoint)

        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("Loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))

    if args.crf:
      model = nn.Sequential(
          model,
          CRF(n_spatial_dims=2)
      )

    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            args.gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            args.gpu_nums = 1
            print("Using single GPU")
            model = model.cuda()  # 1-card data parallel

    model.eval()

    return model, cuda

if __name__ == '__main__':
    
    args = parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    start = time.time()

    model, cuda = get_model(args)

    mean = [123.675, 116.28, 103.53] # ImageNet stats
    std = [58.395, 57.12, 57.375]

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    PILImage = transforms.ToPILImage()
    tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean, std)
    resize = transforms.Resize(input_size, interpolation=2)
    resize_nearest = transforms.Resize(input_size, interpolation=0)

    cmap = colormap()

    i = 0

    print("\n=====> Processing input data")

    if os.path.isdir(args.input):
      glob_path = os.path.join(args.input, '*.png')

      images = glob.glob(glob_path)
      n = len(images)
      
      print("Found {} images".format(n))

      for file_path in images:

        _, file_name = os.path.split(file_path)

        x = cv2.imread(file_path, cv2.IMREAD_COLOR)
        x = x[:, :, ::-1] # bgr -> rgb
        x = PILImage(x)
        x = resize(x)
        x = tensor(x)
        # x = norm(x)
        # x = np.asarray(x, np.float32)
        x = torch.unsqueeze(x, dim=0)

        with torch.no_grad():
            # input_var = Variable(input).cuda()

            if cuda:
              x = x.cuda()

            y_ = model(x)
            y_ = torch.argmax(y_, 1).detach().cpu()

            y_ = torch.squeeze(y_, dim=0)
            y_ = np.asarray(y_, np.uint8)
            y_ = cmap[y_]

        output_path = os.path.join(args.output, file_name)
        cv2.imwrite(output_path, y_)

        i += 1

        print("{}/{} {}".format(i, n, output_path))

    end = time.time()

    print("\n=====> End of demo")

    elapsed = end - start
    hour = elapsed // 3600
    minute = (elapsed % 3600) // 60
    second = int(elapsed - hour * 3600 - minute * 60)
    print("Demo time: {} hours {} minutes {} seconds".format(hour, minute, second))

    if i != 0:
      print("Average time per image: {:.2f} seconds".format(elapsed / i))
