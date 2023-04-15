import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

#  初始化每个类的数目
background_num = 0
building_num = 0
road_num = 0
water_num = 0
barren_num = 0
forest_num = 0
agriculture_num = 0

# label_paths_1 = glob.glob('../DATASET/Original_LoveDA/val/Rural/masks_png_convert/*.png')
label_paths_2 = glob.glob('../DATASET/Original_LoveDA/val/Urban/masks_png_convert/*.png')
label_paths = [label_paths_2]

for label_path in label_paths:
    for file_path in label_path:
        label = Image.open(file_path)
        label = np.array(label)
        background_num += np.sum(label == 0)
        building_num += np.sum(label == 1)
        road_num += np.sum(label == 2)
        water_num += np.sum(label == 3)
        barren_num += np.sum(label == 4)
        forest_num += np.sum(label == 5)
        agriculture_num += np.sum(label == 6)

classes = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture')
numbers = [background_num, building_num, road_num, water_num, barren_num, forest_num, agriculture_num]

print(numbers)
plt.barh(classes, numbers)
plt.title('val_Urban')
plt.savefig("val_Urban.png", dpi = 300, bbox_inches="tight")
# plt.show()