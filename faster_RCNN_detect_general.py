import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
import yaml



if __name__ == '__main__':

    DIR_INPUT = "Traffic_sign_2"
    os.chdir(DIR_INPUT)
    # //////////////////////////  Read the yaml file /////////////////////////////
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    IMG_SIZE = config["IMG_SIZE"]

    # /////////////////////////////////////// Load the model ////////////////////////////////////////////
    trained_model = config["DIR_OUT"]+"/fasterrcnn_resnet50_fpn.pth"
    num_classes = config["CLASS_NUM"]

    # load the base Architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(trained_model))
    model.eval()

    #//////////////////////  Use the trained model to predict for validation dataset /////////////////////////
    # Image to Tensor
    trf = T.Compose([
          #T.Resize(IMG_SIZE),
          #     T.CenterCrop(IMG_SIZE), # make square image
          T.ToTensor(),
          #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = [Image.open(config["TEST_IMAGE"]+"/"+item) for item in os.listdir(config["TEST_IMAGE"]) if item.endswith("jpg")]

    # img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))

    for img in images:
        input_img= trf(img).unsqueeze(0)
        out = model(input_img)

        for item in out:
            #Convert tensor to numpy
            boxes = item['boxes'].detach().numpy().astype('int')
            labels = item['labels'].detach().numpy()
            scores = item['scores'].detach().numpy()
            img = np.array(img)
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))

            for box,label,score in zip(boxes,labels,scores):
                cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 3)
                text = str(label)+">"+str(round(score,2))
                img = cv2.putText(img, text, (box[0], box[1]-25),  cv2.FONT_HERSHEY_SIMPLEX ,
                                    1.5, (0, 0, 255), 2, cv2.LINE_AA)

            ax.set_axis_off()
            ax.imshow(img)

            plt.show()
