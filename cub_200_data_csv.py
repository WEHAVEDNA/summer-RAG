import os
import pandas as pd

def generate_csv(root_dir, output_csv):
    file_dir = (root_dir + '/CUB_200_2011')
    images_file = os.path.join(file_dir, 'images.txt')
    labels_file = os.path.join(file_dir, 'image_class_labels.txt')
    train_test_split_file = os.path.join(file_dir, 'train_test_split.txt')

    csv_dir = os.path.join(root_dir, output_csv)
    
    image_paths = []
    labels = []
    train_test_split = []
    
    with open(images_file, 'r') as f:
        for line in f.readlines():
            image_id, image_path = line.strip().split()
            image_paths.append('images/' + image_path)
    
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            image_id, class_id = line.strip().split()
            labels.append(int(class_id))
    
    with open(train_test_split_file, 'r') as f:
        for line in f.readlines():
            image_id, is_train = line.strip().split()
            train_test_split.append(int(is_train))
    
    data = {
        'image_path': image_paths,
        'label': labels,
        'is_train': train_test_split
    }
    df = pd.DataFrame(data)

    df.to_csv(csv_dir, index=False)
    print(f"CSV file saved to {csv_dir}")

root_dir = (os.path.dirname(os.path.abspath(__file__)) + '/CUB_200_2011')
print(root_dir)
output_csv = 'cub_200_2011_dataset.csv'
generate_csv(root_dir, output_csv)
