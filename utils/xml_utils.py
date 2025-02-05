import xml.etree.ElementTree as ET

def read_xml(xml_path,class_name,eval_mode):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    image_name = root.find('filename')
    
    size = root.find('size')
    image_width = float(size.find('width').text)
    image_height = float(size.find('height').text)
    bboxes = []
    classes = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult')
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        bbox_xmin = float(bbox.find('xmin').text.split('.')[0])
        bbox_ymin = float(bbox.find('ymin').text.split('.')[0])
        bbox_xmax = float(bbox.find('xmax').text.split('.')[0])
        bbox_ymax = float(bbox.find('ymax').text.split('.')[0])
        if difficult is not None:
            difficult = int(obj.find('difficult').text)
            if((difficult == 1) and (eval_mode == 1)):
                continue
        
        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    return image_width, image_height, bboxes, classes
