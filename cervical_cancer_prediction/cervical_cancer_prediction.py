import os, random, shutil, cv2, torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy

# Copy the original data into test, train, and validate folders

path_datasets = './datasets/original_data'
train_dataset = './datasets/train'
test_dataset = './datasets/test'
validate_dataset = './datasets/validate'

split= 0.7

data_dirs = [
    d for d in os.listdir(path_datasets)
    if d.startswith('cervix') and os.path.isdir(os.path.join(path_datasets, d))
]

for cervix_ds in data_dirs:
    # Check to see if files have already been copied
    if os.listdir(os.path.join(train_dataset, cervix_ds)):
        break

    path_cervix_ds = os.path.join(path_datasets, cervix_ds)
    files = [
        f for f in os.listdir(path_cervix_ds)
        if os.path.isfile(os.path.join(path_cervix_ds, f))
    ]

    num_sample = int(len(files) * split)
    training_files = random.sample(files, num_sample)

    for source_file in training_files:
        source_file_path = os.path.join(path_cervix_ds, source_file)
        destination_dir = os.path.join(train_dataset, cervix_ds)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        shutil.copy2(source_file_path, destination_dir)

    files = [
        f for f in os.listdir(path_cervix_ds)
        if f not in training_files
    ]

    num_sample = int(len(files) * 0.5)
    testing_files = random.sample(files, num_sample)

    for source_file in testing_files:
        source_file_path = os.path.join(path_cervix_ds, source_file)
        destination_dir = os.path.join(test_dataset, cervix_ds)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        shutil.copy2(source_file_path, destination_dir)

    validation_files = [
        f for f in os.listdir(path_cervix_ds)
        if f not in training_files and f not in testing_files
    ]

    for source_file in validation_files:
        source_file_path = os.path.join(path_cervix_ds, source_file)
        destination_dir = os.path.join(validate_dataset, cervix_ds)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        shutil.copy2(source_file_path, destination_dir)

cancer_type = ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi']

train_dataset_paths = [
    os.path.join(train_dataset, ct) for ct in cancer_type
]

test_dataset_paths = [
    os.path.join(test_dataset, ct) for ct in cancer_type
]

validation_dataset_paths = [
    os.path.join(validate_dataset, ct) for ct in cancer_type
]

# Training data before preprocessing
i = 0
data_bptr = []

for train_dp in train_dataset_paths:
    for img_file in os.listdir(train_dp):
        img = cv2.imread(os.path.join(train_dp, img_file))
        data_bptr.append((img, i))

    i = i + 1

# Test data before preprocessing
i = 0
data_bpt = []

for test_dp in test_dataset_paths:
    for img_file in os.listdir(test_dp):
        img = cv2.imread(os.path.join(test_dp, img_file))
        data_bpt.append((img,i))

    i = i + 1

# Validation data before preprocessing
i = 0
data_bpv = []

for val_dp in validation_dataset_paths:
    for img_file in os.listdir(val_dp):
        img = cv2.imread(os.path.join(val_dp, img_file))
        data_bpv.append((img,i))

    i = i + 1


# Classes for training, test, and validation data. This is so that we can use useful features
#   (e.g., batching, shuffling) and to easily apply transformations on data
class CancerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elm = self.transform(self.data[idx])
        label = self.labels[idx]

        return elm, label

# Preprocessed data of Dataset type

# Unpacking each pair of images and labels
# [(img,label),...] --> ((img,...), (label,...)) --> (img,...) and (label,...)
training_data = CancerDataset(*zip(*data_bptr))
test_data = CancerDataset(*zip(*data_bpt))
validation_data = CancerDataset(*zip(*data_bpv))

# Loads 8 training data randomly from training_data
# batch_size of 8 is chosen since image size is 3x512x512 which will be computationally expensive and
#   some GPUs wouldn't be able to handle that much load
training_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

# There's no need to shuffle the test and validation data
test_dataloader = DataLoader(test_data, batch_size=8)
validation_dataloader = DataLoader(validation_data, batch_size=8)

# CNN model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1st layer - in: RGB (3 channels), out: 32 channels
        #   ---> 5th layer - in: 215 channels, out: 512 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # Max pooling and this halves the input size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: output features connected to all input features
        # Input features: 512*16*16 ---> Output features: 5
        self.fc1 = nn.Linear(in_features=512*16*16, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=5)

    def forward(self, x):
        # Pool after every convolution
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.pool(f.relu(self.conv3(x)))
        x = self.pool(f.relu(self.conv4(x)))
        x = self.pool(f.relu(self.conv5(x)))

        # Flatten output
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))

        return x

my_model = CNN()

resnet50_cdt = models.resnet50()

# Loss function for multiclass classification
loss_function = nn.CrossEntropyLoss()
# Standard adam optimizer with learning rate of 1e-4
optimizer = torch.optim.Adam(resnet50_cdt.parameters(), lr=1e-4)
#optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)
# Set device as gpu if it exists
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(num_epochs, model_type):
    for epoch in range(num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1,num_epochs))

        # Set model to training mode
        model_type.train()
        for batch_idx, (data, labels) in enumerate(tqdm(training_dataloader)):
            # If cuda exists, data will be moved to the GPU memory
            data = data.to(device)
            outputs = model_type(data)

            # Calculate loss
            loss = loss_function(outputs, labels)
            # Set gradient to 0 before backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()

EPOCHS = 50

train(EPOCHS, resnet50_cdt)
#train(EPOCHS, my_model)

def evaluate(model_type):
    # Set model to evaluation mode
    model_type.eval()

    # Precision: How many were positive (correctly identifying cancer type) out of all predicted positives
    # Recall: How many were correctly positive out of all positives
    # Accuracy: How many did the model correctly predict
    precision_metric = MulticlassPrecision(num_classes=5)
    recall_metric = MulticlassRecall(num_classes=5)
    accuracy_metric = MulticlassAccuracy(num_classes=5)

    # Disable gradient tracking
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model_type(inputs)

            # Calculate precision, recall, and accuracy as we iterate through the test dataloader
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            accuracy_metric.update(outputs, labels)

    # Compute final precision, recall, and accuracy
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    accuracy = accuracy_metric.compute()
    print("Precision: {}, Recall: {}, Accuracy: {}".format(i + 1, precision, recall, accuracy))

evaluate(resnet50_cdt)
#evaluate(my_model)

torch.save(resnet50_cdt.state_dict(), "./resnet50_cdt.pth")