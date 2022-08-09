import json
import cv2
import os
import matplotlib.pyplot as plt
import re


input_path = "/mnt/c/Users/rache/Dropbox/Doctoral Projects/CV4Ecology 2022/Final Images/val"
output_path = "/mnt/c/Users/rache/Dropbox/Doctoral Projects/CV4Ecology 2022/Final Images/Yolov5 Format/val/"

f = open('val_annotations.json')
data = json.load(f)
f.close()

# Helper Functions
def load_images_from_folder(folder):

  file_names = []
  count = 0
  for filename in os.listdir(folder):
        # Remove ".png" from filename
        filename1 =  re.sub('.png', '', filename, count=1)
        img = cv2.imread(os.path.join(folder,filename))
        # Here you can also use shutil
        # Example: dest = shutil.move(source, destination)
        cv2.imwrite(f"{output_path}images/{filename1}.png", img)
        file_names.append(filename)
        count += 1
        
  return file_names

def get_img_ann(image_id):
   img_ann = []
   isFound = False
   for ann in data['annotations']:
       if ann['image_id'] == image_id:
           img_ann.append(ann)
           isFound = True
   if isFound:
       return img_ann
   else:
       return None


def get_img(filename):
  for img in data['images']:
    if img['file_name'] == filename:
      return img


# Processing Images
file_names = load_images_from_folder(f'{input_path}')

# Applying Conversion
count = 0
for filename in file_names:
  # Extracting image 
  img = get_img(filename)
  img_id = img['id']
  img_w = img['width']
  img_h = img['height']

  # Get Annotations for this image
  img_ann = get_img_ann(img_id)

  if img_ann:
    # Remove ".png" from filename
    filename1 =  re.sub('.png', '', filename, count=1)

    # Opening file for current image
    file_object = open(f"{output_path}labels/{filename1}.txt", "a")

    for ann in img_ann:
      current_category = ann['category_id'] - 1 # Subtracting 1 from category id because class labels in yolo format starts from 0.
      current_bbox = ann['bbox']
      x = current_bbox[0]
      y = current_bbox[1]
      w = current_bbox[2]
      h = current_bbox[3]
      
      # Finding midpoints
      x_centre = (x + (x+w))/2
      y_centre = (y + (y+h))/2
      
      # Normalization
      x_centre = x_centre / img_w
      y_centre = y_centre / img_h
      w = w / img_w
      h = h / img_h
      
      # Limiting upto fix number of decimal places
      x_centre = format(x_centre, '.6f')
      y_centre = format(y_centre, '.6f')
      w = format(w, '.6f')
      h = format(h, '.6f')
          
      # Writing current object 
      file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

    file_object.close()
  count += 1