import json
import os
import cv2
import numpy as np
import pandas as pd


with open("/Users/kovidsharma/Desktop/BTP/pycharm/macvi/Mods/mods.json") as file:
    data = json.load(file)["dataset"]

file.close()


def save_figure(image, save_dir):
    cv2.imwrite(save_dir, image)


def create_mask(image, water_edges, thickness):

    mask_ = np.zeros(shape=image.shape)
    for edge in water_edges:
        x_axs = np.array(edge['x_axis'])
        y_axs = np.array(edge['y_axis'])
        for i in range(len(x_axs) - 1):
            start_point = (x_axs[i], y_axs[i])
            end_point = (x_axs[i + 1], y_axs[i + 1])
            clr = (255, 255, 255)
            cv2.line(mask_, start_point, end_point, clr, thickness=thickness)

    return mask_


def bboxes(obstacles):
    lis = []
    for obstacle in obstacles:
        obj_type = obstacle["type"]
        bbox = obstacle["bbox"]
        lis1 = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        if obj_type == "person":
            lis1.append(0)
        elif obj_type == "ship":
            lis1.append(1)
        else:
            lis1.append(2)

        lis.append(lis1)

    return lis


meta_data = {
    "save_img_pth": [],
    "save_mask_pth": [],
    "roll": [],
    "pitch": [],
    "yaw": [],
    "latitude": [],
    "longitude": [],
    "no_of_bbxs": [],
    "bounding_boxes": []
}

seq_folder = "/Users/kovidsharma/Desktop/BTP/pycharm/macvi/Mods/sequences"
data_dir = "/Code/data"

count = 0
for seq_id in range(data["num_seq"]):
    seq = data["sequences"][seq_id]
    seq_pth = seq["path"]
    for frame_id in range(seq["num_frames"]):
        frame = seq["frames"][frame_id]
        img_file_name = frame["image_file_name"]
        pth = os.path.join(seq_pth, img_file_name)
        roll, pitch, yaw = frame["roll"], frame["pitch"], frame["yaw"]
        latitude, longitude = frame["latitude"], frame["longitude"]
        img_pth = seq_folder + pth
        img = cv2.imread(img_pth)
        wtedges = frame["water_edges"]
        obs = frame["obstacles"]
        bbox_lis = bboxes(obs)
        save_img_pth = f"/images/image_{count}.jpg"
        save_mask_pth = f"/masks/mask_{count}.jpg"
        mask = create_mask(img, wtedges, thickness=5)
        save_figure(img, save_dir=f"{data_dir}{save_img_pth}")
        save_figure(mask, save_dir=f"{data_dir}{save_mask_pth}")
        meta_data["save_img_pth"].append(save_img_pth)
        meta_data["save_mask_pth"].append(save_mask_pth)
        meta_data["roll"].append(roll)
        meta_data["pitch"].append(pitch)
        meta_data["yaw"].append(yaw)
        meta_data["latitude"].append(latitude)
        meta_data["longitude"].append(longitude)
        meta_data["no_of_bbxs"].append(len(bbox_lis))
        meta_data["bounding_boxes"].append(bbox_lis)
        count += 1

    print("True")

df = pd.DataFrame.from_dict(meta_data)
df.to_csv("metadata.csv", index=False)








