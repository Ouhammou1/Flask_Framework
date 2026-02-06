import os
import shutil

DATA_ROOT = "data"

# Top-level dirs to delete
DELETE_TOP = [
    "dummy",
    "test_voc",
    "VOC2012_test",
    "VOC2012_train_val",
]

# Inside VOC2012, keep ONLY these
KEEP_VOC = {
    "JPEGImages",
    "SegmentationClass",
    "ImageSets",
}

# Inside ImageSets, keep ONLY this
KEEP_IMAGESETS = {
    "Segmentation",
}

def delete_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {path}")

def main():
    # Delete top-level unused dirs
    for d in DELETE_TOP:
        delete_path(os.path.join(DATA_ROOT, d))

    voc_root = os.path.join(DATA_ROOT, "VOCdevkit", "VOC2012")

    # Clean VOC2012
    for item in os.listdir(voc_root):
        path = os.path.join(voc_root, item)
        if item not in KEEP_VOC:
            delete_path(path)

    # Clean ImageSets
    imagesets = os.path.join(voc_root, "ImageSets")
    for item in os.listdir(imagesets):
        path = os.path.join(imagesets, item)
        if item not in KEEP_IMAGESETS:
            delete_path(path)

    print("\n✅ Cleanup finished. Dataset ready for training.")

if __name__ == "__main__":
    main()

