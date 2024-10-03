import json
import cv2

with open("./Mods/mods.json", 'r') as file:
    data = json.load(file)['dataset']

main_dir_path = "/Users/kovidsharma/Desktop/BTP/pycharm/macvi/Mods/sequences"
image_folder = "/Users/kovidsharma/Desktop/BTP/pycharm/macvi/images"
labels_folder = "/Users/kovidsharma/Desktop/BTP/pycharm/macvi/masks"
prac_folder = "/Users/kovidsharma/Desktop/BTP/pycharm/macvi/img_with_annotations/"
print(data['sequences'][0].keys())
print(data['sequences'][0]['path'])
print(len(data['sequences'][0]['frames']))
print(data['sequences'][0]['frames'][0].keys())


def draw_water_edges(image, water_edges):
    for edge in water_edges:
        x_axs = edge['x_axis']
        y_axs = edge['y_axis']
        for i in range(len(x_axs) - 1):
            cv2.line(image, (x_axs[i], y_axs[i]), (x_axs[i + 1], y_axs[i + 1]), (255, 0, 0), 2)


def draw_bounding_box(image, top_left, bottom_right, color, thickness=2):

    cv2.rectangle(image, top_left, bottom_right, color, thickness)


def draw_bounding_boxes(image, obstacles, thickness=2):
    for obstacle in obstacles:
        obj_type = obstacle['type']
        bbox = obstacle['bbox']

        if obj_type == "person":
            color = (0, 255, 0)

        elif obj_type == "ship":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        top_left = (int(bbox[0] - 0.5*bbox[2]), int(bbox[1] - 0.5*bbox[3]))
        top_right = (int(bbox[0] + 0.5*bbox[2]), int(bbox[1] + 0.5*bbox[3]))
        draw_bounding_box(image, top_left, top_right, color, thickness=thickness)


def save_figure(image, file_name, save_dir):
    cv2.imwrite(f"{save_dir}{file_name}", image)


count = 1
for seq in data['sequences']:
    seq_pth = seq['path']
    for frame in seq['frames']:
        img_file_name = frame['image_file_name']
        obstacles = frame['obstacles']
        water_edges = frame['water_edges']
        final_pth = f"{main_dir_path}{seq_pth}{img_file_name}"
        img = cv2.imread(final_pth)
        draw_water_edges(img, water_edges)
        draw_bounding_boxes(img, obstacles)
        save_figure(img, f"img_annot_{count}.jpg", prac_folder)
        count += 1







