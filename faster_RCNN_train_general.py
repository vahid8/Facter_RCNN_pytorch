
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

from tqdm import tqdm

import yaml




class TrainDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        l = records.label.tolist()
        l = np.array(l)
        labels = torch.as_tensor(l, dtype=torch.int64)
        #labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes

        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            #target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()
            target['boxes'] = torch.tensor(sample['bboxes'],dtype=torch.float32)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# Albumentations
def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE,IMG_SIZE),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        A.Resize(IMG_SIZE,IMG_SIZE),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))

def PrepareData(csv_path:str)->(pd.DataFrame,pd.DataFrame):

    train_df = pd.read_csv(csv_path)
    print(train_df.head())

    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    def expand_bbox(x):
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(float)
    train_df['y'] = train_df['y'].astype(float)
    train_df['w'] = train_df['w'].astype(float)
    train_df['h'] = train_df['h'].astype(float)

    image_ids = train_df['image_id'].unique()
    valid_ids = image_ids[:]
    train_ids = image_ids[:]

    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    return train_df,valid_df


if __name__ == '__main__':

    DIR_INPUT =  "Traffic_sign_2"
    os.chdir(DIR_INPUT)
    #//////////////////////////  Read the yaml file /////////////////////////////
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    IMG_SIZE = config["IMG_SIZE"]

    train_df,valid_df = PrepareData(config["TRAIN_LABLE"])

    print("valid_df.shape:{}".format(valid_df.shape))
    print("train_df.shape:{}".format(train_df.shape))


    # Bring the data in the required foemat by pytorch as a class
    train_dataset = TrainDataset(train_df, config["TRAIN_IMAGE"], get_train_transform())
    valid_dataset = TrainDataset(valid_df, config["TRAIN_IMAGE"], get_valid_transform())


    # Load dataset to the torch using its DataLoader function

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Select the device to train on
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,  config["CLASS_NUM"])

    #/////////////////// Show one sample of training data
    images, targets, image_ids = next(iter(train_data_loader))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(sample)
    plt.show()
    #///////////////////////////////////////////////////////

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None



    loss_hist = Averager()
    itr = 1
    checkpoint_num = 0

    for epoch in range( config["EPOCHS_NUM"]):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader,desc="epoch "+str(epoch)+"/"+str( config["EPOCHS_NUM"])):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            #if itr % 50 == 0:
            #    print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()


        if epoch % 10 == 0:     # Save checkpoint each --- iteration
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_hist.value,
            }, config["DIR_OUT"]+"/checkpoint_"+str(checkpoint_num)+".pt")
            checkpoint_num += 1

        print(f"Epoch #{epoch} loss: {loss_hist.value}")


    #Save the model
    torch.save(model.state_dict(), config["DIR_OUT"]+'/fasterrcnn_resnet50_fpn.pth')
    evaluate(model, valid_data_loader, device=device)



