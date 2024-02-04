import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import EfficientFace
from PIL import Image
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def main():
    # chinh ve 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # create model
    ## EfficientFace
    model = EfficientFace.efficient_face()
    model.fc = nn.Linear(1024, 7)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(
        '..\checkpoint\pv2_lowl2-model_best.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict(pre_trained_dict)

    normalize = transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                     std=[0.20735591, 0.18981615, 0.18132027])

    transforms_com = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    model.eval()
    img = r"..\DATASET\test\0\test_2403_aligned.jpg"
    img = Image.open(img).convert('RGB')
    img_t = transforms_com(img)
    img_t = img_t.unsqueeze(0)
    img_t = img_t.cuda()

    labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    output = model(img_t)
    output = output.detach().cpu().numpy()[0]
    target_layers = [model.module.conv5]
    cam = GradCAM(model=model.module, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_t, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.asarray(img.resize((224, 224))) / 255,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(labels[np.argmax(output)])
    plt.show()


if __name__ == '__main__':
    main()