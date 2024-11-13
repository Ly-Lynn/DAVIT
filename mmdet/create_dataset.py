import json
import shutil
import os
from sklearn.model_selection import train_test_split

def split_coco_dataset(annotations_file, images_dir, output_dir, val_split=0.2, random_state=42):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for d in [train_dir, val_dir]:
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
    
    # Load annotations file
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get list of image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Split image IDs into train and validation sets
    train_img_ids, val_img_ids = train_test_split(
        image_ids,
        test_size=val_split,
        random_state=random_state
    )
    
    # Create train and validation annotation files
    train_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': [],
        'annotations': []
    }
    
    val_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': [],
        'annotations': []
    }
    
    # Split images
    for img in coco_data['images']:
        if img['id'] in train_img_ids:
            train_data['images'].append(img)
            src_path = os.path.join(images_dir, img['file_name'])
            dst_path = os.path.join(train_dir, 'images', img['file_name'])
            shutil.copy2(src_path, dst_path)
        else:
            val_data['images'].append(img)
            src_path = os.path.join(images_dir, img['file_name'])
            dst_path = os.path.join(val_dir, 'images', img['file_name'])
            shutil.copy2(src_path, dst_path)
    
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_img_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    
    # Save annotation files
    with open(os.path.join(train_dir, 'annotations.json'), 'w') as f:
        json.dump(train_data, f)
    
    with open(os.path.join(val_dir, 'annotations.json'), 'w') as f:
        json.dump(val_data, f)
    
    # Print statistics
    print(f"Total images: {len(image_ids)}")
    print(f"Training images: {len(train_img_ids)}")
    print(f"Validation images: {len(val_img_ids)}")
    print(f"Training annotations: {len(train_data['annotations'])}")
    print(f"Validation annotations: {len(val_data['annotations'])}")

# Example usage
if __name__ == "__main__":
    annotations_file = r"D:\codePJ\BKAI\Track1\data\Dark Vehicle Detection\train\_annotations.coco.json"
    images_dir = r"D:\codePJ\BKAI\Track1\data\Dark Vehicle Detection\train"
    output_dir = r"D:\codePJ\BKAI\Track1\davit\mmdet\data\coco"
    
    split_coco_dataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_dir=output_dir,
        val_split=0.2
    )