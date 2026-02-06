import os
import cv2
import numpy as np

ROOT = "/home/bouhammo/Home/pascal_voc_segmentation/pro1/data/VOCdevkit/VOC2012" 
IMG_DIR= os.path.join(ROOT  , "JPEGImages")
MASK_DIR = os.path.join(ROOT, "SegmentationClass")
SPLIT_DIR = os.path.join(ROOT, "ImageSets/Segmentation")

VALID_CLASSES = set(range(0,21)) |  {255}

print(VALID_CLASSES)



def read_ids(split):
    path=os.path.join(SPLIT_DIR , f"{split}.txt")
    with open(path) as f:
        ids =[]
        for line in f:
            clean = line.strip()
            if clean:
                ids.append(clean)
        return ids




def check_pair(img_id):
    img_path  = os.path.join(IMG_DIR , img_id + ".jpg")
    mask_path = os.path.join(MASK_DIR , img_id + ".png")

    if not  os.path.exists(img_path):
        return f"❌ Missing image: {img_id}.jpg"

    if not os.path.exists(mask_path):
        return f"❌ Missing mask: {img_id}.png"
    
    img  =cv2.imread(img_path)
    mask = cv2.imread(mask_path , cv2.IMREAD_GRAYSCALE)
    print(mask)

    if img is None:
        return f"❌ Corrupted image: {img_id}.jpg"
    if mask is None:
        return f"❌ Corrupted mask: {img_id}.png"
    
    if img.shape[:2] != mask.shape:
        return f"❌ Size mismatch: {img_id} image {img.shape[:2]} vs mask {mask.shape}"
    values = set(np.unique(mask))
    print(values)
    bad = values  - VALID_CLASSES
    if bad:
        return f"❌ Invalid labels {bad} in mask {img_id}.png"
    return None







def    clean_split(split):
    print(f"checking {split} split")
    ids = read_ids(split)
    
    valid_ids = []

    for img_id in ids:
        error = check_pair(img_id)
        if error:
            print(error)
        else:
            valid_ids.append(img_id)
    
    path = os.path.join(SPLIT_DIR , f"{split}.txt")
    with  open(path , "w") as f:
        for i in valid_ids:
            f.write(i , "\n")
    
    print(f"✅ {len(valid_ids)} / {len(ids)} samples valid")




            
def main():
    for split in ["train", "val", "test"]:
        split_file = os.path.join(SPLIT_DIR , f"{split}.txt")
        if os.path.exists(split_file):
            clean_split(split)
        print(split_file)



if __name__ == "__main__":
    main()