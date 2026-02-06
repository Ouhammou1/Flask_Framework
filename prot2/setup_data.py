"""
Dataset setup and verification
"""
import os
import zipfile
import subprocess
import sys
import shutil
from pathlib import Path
import requests
import tqdm

class DatasetSetup:
    """Handle dataset downloading and setup"""
    
    KAGGLE_URL = "https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset"
    DATASET_PATHS = [
        "data/VOCdevkit/VOC2012",
        "VOCdevkit/VOC2012",
        "data/VOC2012",
        "VOC2012",
        "data/VOC2012_train_val/VOCdevkit/VOC2012"
    ]
    
    @classmethod
    def find_dataset(cls):
        """Find existing dataset"""
        for path_str in cls.DATASET_PATHS:
            path = Path(path_str)
            if path.exists():
                print(f"✅ Found dataset at: {path.absolute()}")
                return path
        return None
    
    @classmethod
    def verify_dataset(cls, dataset_path):
        """Verify dataset structure"""
        print(f"\n🔍 Verifying dataset at: {dataset_path}")
        
        required = {
            "JPEGImages": "*.jpg",
            "SegmentationClass": "*.png",
            "ImageSets/Segmentation": "*.txt"
        }
        
        all_good = True
        for dir_name, pattern in required.items():
            dir_path = dataset_path / dir_name
            
            if not dir_path.exists():
                print(f"❌ Missing directory: {dir_name}")
                all_good = False
                continue
            
            # Count files
            try:
                if pattern == "*.jpg":
                    files = list(dir_path.glob("*.jpg"))
                    print(f"   {dir_name}: {len(files):,} images")
                elif pattern == "*.png":
                    files = list(dir_path.glob("*.png"))
                    print(f"   {dir_name}: {len(files):,} masks")
                elif pattern == "*.txt":
                    txt_files = list(dir_path.glob("*.txt"))
                    for txt_file in txt_files:
                        with open(txt_file, 'r') as f:
                            lines = len([line for line in f if line.strip()])
                        print(f"   {txt_file.name}: {lines:,} entries")
            except Exception as e:
                print(f"   Error reading {dir_name}: {e}")
                all_good = False
        
        return all_good
    
    @classmethod
    def download_via_kaggle(cls):
        """Download dataset using Kaggle API"""
        print("\n📥 Downloading via Kaggle API...")
        
        try:
            # Check if kaggle is installed
            subprocess.run(["kaggle", "--version"], 
                         capture_output=True, check=True)
            
            # Download dataset
            print("Downloading (this may take several minutes)...")
            result = subprocess.run([
                "kaggle", "datasets", "download",
                "gopalbhattrai/pascal-voc-2012-dataset"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Unzip
                zip_path = "pascal-voc-2012-dataset.zip"
                if os.path.exists(zip_path):
                    print("📂 Extracting dataset...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        total = len(zip_ref.namelist())
                        for i, file in tqdm.tqdm(enumerate(zip_ref.namelist()), 
                                               total=total, desc="Extracting"):
                            zip_ref.extract(file, "data")
                    
                    os.remove(zip_path)
                    print("✅ Download complete!")
                    return True
            else:
                print(f"❌ Kaggle error: {result.stderr}")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Kaggle CLI not found or not configured")
            return False
    
    @classmethod
    def manual_download_instructions(cls):
        """Show manual download instructions"""
        instructions = f"""
        📝 MANUAL DOWNLOAD REQUIRED
        
        1. Visit: {cls.KAGGLE_URL}
        2. Click the 'Download' button (requires Kaggle login)
        3. Save the file as 'pascal-voc-2012-dataset.zip'
        4. Place it in this directory: {Path.cwd()}
        5. Run this script again
        
        Or extract manually:
        mkdir -p data
        unzip pascal-voc-2012-dataset.zip -d data/
        
        Expected structure after extraction:
        {Path.cwd()}/data/VOCdevkit/VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/
        """
        print(instructions)
        return False
    
    @classmethod
    def setup(cls):
        """Main setup function"""
        print("=" * 60)
        print("PASCAL VOC 2012 DATASET SETUP")
        print("=" * 60)
        
        # Check if dataset already exists
        dataset_path = cls.find_dataset()
        if dataset_path:
            if cls.verify_dataset(dataset_path):
                return dataset_path
            else:
                print("\n⚠️  Dataset exists but is incomplete")
        
        # Try to download
        print("\n📥 Dataset not found. Downloading...")
        
        # Try Kaggle API first
        if cls.download_via_kaggle():
            dataset_path = cls.find_dataset()
            if dataset_path and cls.verify_dataset(dataset_path):
                return dataset_path
        
        # Fall back to manual instructions
        cls.manual_download_instructions()
        
        # Check if user manually downloaded
        zip_path = Path("pascal-voc-2012-dataset.zip")
        if zip_path.exists():
            print("\nFound zip file. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data")
            zip_path.unlink()
            
            dataset_path = cls.find_dataset()
            if dataset_path:
                return dataset_path
        
        return None
    
    @classmethod
    def create_test_dataset(cls):
        """Create a small test dataset for development"""
        print("\n🛠️  Creating test dataset...")
        
        test_dir = Path("data/test_voc")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        (test_dir / "JPEGImages").mkdir(exist_ok=True)
        (test_dir / "SegmentationClass").mkdir(exist_ok=True)
        (test_dir / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
        
        # Create 10 test images
        import numpy as np
        from PIL import Image
        
        for i in range(10):
            # Create random image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(test_dir / "JPEGImages" / f"test_{i:06d}.jpg")
            
            # Create random mask (4 classes)
            mask_array = np.random.randint(0, 4, (256, 256), dtype=np.uint8) * 50
            mask = Image.fromarray(mask_array, mode='L')
            mask.save(test_dir / "SegmentationClass" / f"test_{i:06d}.png")
        
        # Create splits
        with open(test_dir / "ImageSets" / "Segmentation" / "train.txt", "w") as f:
            f.write("\n".join([f"test_{i:06d}" for i in range(7)]))
        
        with open(test_dir / "ImageSets" / "Segmentation" / "val.txt", "w") as f:
            f.write("\n".join([f"test_{i:06d}" for i in range(7, 10)]))
        
        print(f"✅ Created test dataset at: {test_dir}")
        return test_dir

if __name__ == "__main__":
    dataset_path = DatasetSetup.setup()
    
    if dataset_path:
        print(f"\n🎉 Dataset ready at: {dataset_path.absolute()}")
        print("\nYou can now run:")
        print("python app.py")
    else:
        print("\n⚠️  Using test dataset for development")
        test_path = DatasetSetup.create_test_dataset()
        print(f"Test dataset at: {test_path}")
        print("\nRun with test data:")
        print("python app.py")