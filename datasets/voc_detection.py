import os
import glob
import time

import numpy as np

from PIL import Image

from utils.txt_utils import read_txt
from utils.xml_utils import read_xml
class Merging_Dataset:
    def __init__(self, datasets):
        self.data_dic = {name:dataset for name, dataset in enumerate(datasets)}
        self.datasets = []
        for name in self.data_dic.keys():
            dataset = self.data_dic[name]
            for i in range(len(dataset)):
                self.datasets.append([name, i])

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        name, index = self.datasets[index]
        return self.data_dic[name][index]

class PASCAL_VOC_Dataset:
    def __init__(self, root_dir, domain, transform, class_path, eval_mode, preload=True):
        # Clean up the root_dir path
        if root_dir[-1] == '/':
            root_dir = root_dir[:-1]
        if not os.path.isdir(root_dir):
            raise Exception("There is no dataset_root dir")
        
        self.image_dir = os.path.join(root_dir, f'{domain}/image/')
        self.xml_dir = os.path.join(root_dir, f'{domain}/xml/')
        
        if not os.path.isdir(self.image_dir):
            raise Exception("There is no image dir")
        if not os.path.isdir(self.xml_dir):
            raise Exception("There is no xml dir")
        
        self.eval_mode = eval_mode
        self.class_names = read_txt(class_path)
        self.class_dic = {name: index for index, name in enumerate(self.class_names)}
        self.classes = len(self.class_names)
        self.image_ids = [xml_name.replace('.xml', '') for xml_name in os.listdir(self.xml_dir)]
        self.transform = transform
        
        # Set preload flag and initialize cache if needed.
        self.preload = preload


        # if self.preload:
        #     print("Preloading dataset into memory...")
        #     self.data_cache = []
        #     for idx in range(len(self.image_ids)):
        #         img, target = self.load_item(idx)
        #         # Apply the transformation during preload if desired.
        #         if self.transform is not None:
        #             img, target = self.transform(img, target)
        #         self.data_cache.append((img, target))
        #     print("Preloading complete.")

        if self.preload:
            print("Preloading dataset into memory...")
            self.data_cache = []
            start_time = time.time()
            num_items = len(self.image_ids)
            for idx in range(num_items):
                img, target = self.load_item(idx)
                # Apply the transformation during preload if desired.
                if self.transform is not None:
                    img, target = self.transform(img, target)
                self.data_cache.append((img, target))
                
                # Print intermediate throughput every 100 items
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Loaded {idx + 1}/{num_items} items - Throughput: {(idx + 1) / elapsed:.2f} items/sec")
            
            total_time = time.time() - start_time
            print(f"Preloading complete. Loaded {num_items} items in {total_time:.2f} seconds "
                f"({num_items / total_time:.2f} items/sec).")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.preload:
            return self.data_cache[index]
        else:
            img, target = self.load_item(index)
            if self.transform is not None:
                img, target = self.transform(img, target)
            return img, target

    def load_item(self, index):
        """
        Helper method to load a single image and its corresponding target
        from disk. This encapsulates the logic originally in __getitem__.
        """
        image_id = self.image_ids[index]
        possible_img_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.png', '.PNG']
        image = None
        for ext in possible_img_extensions:
            img_path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break
        if image is None:
            raise Exception(f"File not found for {image_id} with supported extensions {possible_img_extensions}!")
        
        target = []
        width, height, bboxes, classes = read_xml(os.path.join(self.xml_dir, image_id + '.xml'), self.class_names, self.eval_mode)
        for (xmin, ymin, xmax, ymax), class_name in zip(bboxes, classes):
            try:
                x1 = float(xmin) / width
                y1 = float(ymin) / height
                x2 = float(xmax) / width
                y2 = float(ymax) / height
            except:
                raise Exception(f"In xml file, width or height value is 0 in {image_id}")
            target.append([x1, y1, x2, y2, self.class_dic[class_name]])
        return image, target


