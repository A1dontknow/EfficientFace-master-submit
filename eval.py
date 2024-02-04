# Description: Evaluate the model on the test set and plot the confusion matrix

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def main():
    print(torch.cuda.get_device_name(0))

    # create model
    ## EfficientFace
    classifier = EfficientFace.efficient_face()
    classifier.fc = nn.Linear(1024, 7)

    classifier = torch.nn.DataParallel(classifier).cuda()
    checkpoint = torch.load(args.checkpoint_path)
    pre_trained_dict = checkpoint['state_dict']
    classifier.load_state_dict(pre_trained_dict)

    # define loss function (criterion) and optimizer
    criterion_val = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'test')

    # RAF-DB
    normalize = transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                      std=[0.20735591, 0.18981615, 0.18132027])


    test_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            normalize]))


    val_loader = torch.utils.data.DataLoader(test_dataset,#test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)


    validate(val_loader, classifier, criterion_val)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)

    # Assuming 'cm' is the confusion matrix
    class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']  # Replace with your actual class names

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap using seaborn
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    # Set labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)

    # Rotate tick labels if needed
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust the layout to make room for tick labels
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from models import EfficientFace
    import os

    from train import validate
    from models import EfficientFace

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DATASET/')
    parser.add_argument('--checkpoint_path', type=str, default=r'checkpoint\pv2_lowl2-model_best.pth.tar')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
    args = parser.parse_args()
    main()