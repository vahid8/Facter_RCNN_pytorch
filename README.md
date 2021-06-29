# Facter_RCNN_pytorch
This program is the implementation of facter RCNN is pytorch with simple steps:
1. cretae a folder like sample (Traffic_sign) containing images (test and train) ,labels (test, train), model and config.yaml
Note: if the labels are in yolo format you can use available script here (yolo to csv) and change the format to pascal.
2. edit the config file with correct paths
3. run the faster_RCNN_train_general.py to train the model.
4. run the faster_RCNN_detect_general.py to use the model for detections.
