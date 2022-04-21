from modules.unet import UNet
# from modules.crf import CRFRNN
import segmentation_models_pytorch as smp

def build_model(model_name, encoder, in_domain, out_domain, num_classes, downsample, crf, device):
    
    if in_domain in ['rgb']:
      in_channels = 3
    else:
      in_channels = 1

    if out_domain in ['normal']:
      out_channels = 3
    elif out_domain in ['seg']:
      out_channels = num_classes
    else:
      out_channels = 1

    print('Model: {}'.format(model_name))
    print('Encoder: {}'.format(encoder))
    print('Depth: {}'.format(downsample))

    if model_name == 'unet':

        if encoder == 'unet':     
            base_model = UNet(downsample=downsample, in_channels=in_channels, out_channels=out_channels)
        elif encoder == 'resnet34':
            base_model = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=in_channels,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channels,                      # model output channels (number of classes in your dataset)
                encoder_depth=downsample
            )

    if crf:
      print('Has CRF')
      base_model = CRFRNN(base_model, device)

    return base_model

