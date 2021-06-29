import pandas as pd
import numpy as np
import os

label_path = "Traffic_sign/labels/train/"
output_path = "Traffic_sign/labels/"

txt_files = [label_path+"/"+item for item in os.listdir(label_path) if item.endswith(".txt")]
IMG_SIZE = 2048
data = list()
for item in txt_files:
    image_id = item.split("/")[-1].split(".")[0]
    with open(item,"r") as f:
        for line in f.readlines():
            if len(line)>1:
                line = line.split()
                label = int(line[0])

                cnt_w_h = [float(item) for item in line[1:5]]
                x = cnt_w_h[0] - cnt_w_h[2]/2
                y = cnt_w_h[1] - cnt_w_h[3]/2
                bbox = [int(x*IMG_SIZE),int(y*IMG_SIZE),int(cnt_w_h[2]*IMG_SIZE),int(cnt_w_h[3]*IMG_SIZE)]
                a = max(bbox)
                if a>IMG_SIZE:
                    print("error ")


                data.append([image_id,bbox,label])


data_df = pd.DataFrame(data,columns = ["image_id","bbox","original_label"])


# Get number of classes
labels = data_df.original_label.tolist()


new_labels =list()
mapping_info=list()
num = 0 # for background
for item in labels:
    if item in mapping_info:
        new_id = mapping_info.index(item)+1 # because the idx starts from 0 and the labels from 1
    else: # it is  a new id
        mapping_info.append(item)
        num+=1 # ids start from one
        new_id = num

    new_labels.append(new_id)

print("number of classes:{}".format(num+1))#one for the background

data_df["label"] = new_labels
print(data_df.head())
data_df.to_csv(output_path+"/train.csv",sep=",", index=False)

# Create mapping info
mapping_df = pd.DataFrame(zip(range(1,len(mapping_info)+1),mapping_info),columns =['idx','label'])
print(mapping_df.head())
mapping_df.to_csv(output_path+"/mapping.csv",sep=",", index=False)

