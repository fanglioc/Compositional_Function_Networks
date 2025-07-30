import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cfn_pytorch.function_nodes import GenericConvNode, ReLUFunctionNode, LinearFunctionNode
from torch.amp import GradScaler, autocast
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 1. Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Enhanced Data Loading and Preprocessing
print("Loading and preprocessing CIFAR-10 data with advanced augmentation...")

# Advanced data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4, 
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)

# 3. ResNet-style Deep CFN Model
class ResidualBlock(nn.Module):
    def __init__(self, input_shape, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = GenericConvNode(input_shape, in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu1 = ReLUFunctionNode(self.conv1.output_dim)
        self.conv2 = GenericConvNode(self.conv1.output_shape_2d[1:], out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = GenericConvNode(input_shape, in_channels, out_channels, kernel_size=1, stride=stride)
            
        self.output_shape_2d = self.conv2.output_shape_2d
        self.output_dim = self.conv2.output_dim

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Main path
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        
        # Shortcut path
        shortcut_out = self.shortcut(x) if isinstance(self.shortcut, GenericConvNode) else x
        
        # Reshape to 2D for residual connection
        out_2d = out.view(batch_size, *self.conv2.output_shape_2d)
        shortcut_out_2d = shortcut_out.view(batch_size, *self.conv2.output_shape_2d)
        
        final_out = nn.functional.relu(out_2d + shortcut_out_2d)
        return final_out.view(batch_size, -1)

class DeepCFN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DeepCFN, self).__init__()
        self.in_channels = 64

        self.conv1 = GenericConvNode((32, 32), 3, 64, kernel_size=3, padding=1)
        self.relu1 = ReLUFunctionNode(self.conv1.output_dim)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], input_shape=self.conv1.output_shape_2d[1:], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], input_shape=self.layer1.output_shape_2d[1:], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], input_shape=self.layer2.output_shape_2d[1:], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], input_shape=self.layer3.output_shape_2d[1:], stride=2)
        
        self.pool = nn.AvgPool2d(4)
        self.classifier = LinearFunctionNode(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, input_shape, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        current_shape = input_shape
        for s in strides:
            b = block(current_shape, self.in_channels, out_channels, s)
            layers.append(b)
            self.in_channels = out_channels
            current_shape = b.output_shape_2d[1:]
        
        layer_module = nn.Sequential(*layers)
        # Manually attach the final output shape to the sequential module
        layer_module.output_shape_2d = (out_channels, *current_shape)
        return layer_module

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.relu1(self.conv1(x))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        batch_size = out.shape[0]
        out = out.view(batch_size, 512, *self.layer4.output_shape_2d[1:])
        out = self.pool(out)
        out = out.view(batch_size, -1)
        
        out = self.classifier(out)
        return out

def ResNet18_CFN():
    return DeepCFN(ResidualBlock, [2, 2, 2, 2])

# Mixup data augmentation
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 4. Model Initialization, Optimizer, and Loss Function
print("Initializing ResNet-style DeepCFN model...")
model = ResNet18_CFN().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
num_epochs = 200

# Cosine annealing scheduler with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Gradient scaler for mixed precision training
scaler = GradScaler()

# 5. Training Loop with optimizations
print(f"Starting training for {num_epochs} epochs...")
best_acc = 0.0
save_path = 'best_cifar10_model_pure.pth'
last_10_accs = []

# Training time tracking
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Adjust mixup alpha based on epoch
    mixup_alpha = min(1.0, 0.2 + epoch * 0.6 / 100)
    
    # Phase 2: Switch to Adam after 150 epochs for fine-tuning
    if epoch == 150:
        print("Switching to Adam optimizer for fine-tuning...")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Apply mixup to entire batch with 70% probability
        use_mixup = np.random.random() < 0.7
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
            
        optimizer.zero_grad()
        
        # Use mixed precision training
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        # Scale gradients and optimize
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, lr: {scheduler.get_last_lr()[0]:.6f}')
            running_loss = 0.0
    
    # Step the scheduler after each epoch
    scheduler.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}/{num_epochs} - Test Accuracy: {accuracy:.2f} %')
    
    # Save the best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), save_path)
        print(f'New best model saved with accuracy: {accuracy:.2f} %')
    
    # Keep track of last 10 accuracies for early stopping check
    last_10_accs.append(accuracy)
    if len(last_10_accs) > 10:
        last_10_accs.pop(0)
    
    # Early stopping check: if we've plateaued in the last 10 epochs and close to best accuracy
    if len(last_10_accs) == 10 and epoch > 120:
        mean_acc = sum(last_10_accs) / 10
        acc_std = np.std(last_10_accs)
        if acc_std < 0.1 and mean_acc > best_acc - 0.5:
            print(f"Accuracy has plateaued at {mean_acc:.2f}% with std {acc_std:.4f}. Early stopping.")
            break

training_time = time.time() - start_time
print(f"Total training time: {training_time/60:.2f} minutes")

# Evaluate with test-time augmentation
print("Evaluating final model with test-time augmentation...")

# Load the best model
model.load_state_dict(torch.load(save_path))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0], data[1]
        labels = labels.to(device)
        
        # Apply TTA with tensor operations directly
        batch_preds = []
        
        # Original images
        images1 = images.to(device)
        # Flatten the input as expected by the model
        images1_flat = images1.view(images1.size(0), -1)
        outputs1 = model(images1_flat)
        batch_preds.append(outputs1.softmax(dim=1))
        
        # Horizontally flipped images
        images2 = torch.flip(images, dims=[3]).to(device)  # Flip along width
        # Flatten the input
        images2_flat = images2.view(images2.size(0), -1)
        outputs2 = model(images2_flat)
        batch_preds.append(outputs2.softmax(dim=1))
        
        # For the crops, we need to handle the different sizes
        # Option 1: Skip the crops since they have different dimensions
        # Option 2: Resize crops back to 32x32 before feeding to model
        
        # Using Option 2: Resize crops to original size
        resize_transform = transforms.Resize((32, 32))
        
        # Define crops
        crop_size = 28  # smaller than 32
        crops_list = [
            # Center crop
            images[:, :, 2:-2, 2:-2],
            # Top-left crop
            images[:, :, :crop_size, :crop_size],
            # Top-right crop
            images[:, :, :crop_size, -crop_size:],
            # Bottom-left crop
            images[:, :, -crop_size:, :crop_size],
            # Bottom-right crop
            images[:, :, -crop_size:, -crop_size:]
        ]
        
        for crop in crops_list:
            # Resize back to 32x32 if needed
            if crop.shape[2] != 32 or crop.shape[3] != 32:
                resized_crop = torch.stack([resize_transform(img) for img in crop])
            else:
                resized_crop = crop
                
            resized_crop = resized_crop.to(device)
            # Flatten the input
            crop_flat = resized_crop.view(resized_crop.size(0), -1)
            outputs_crop = model(crop_flat)
            batch_preds.append(outputs_crop.softmax(dim=1))
        
        # Average predictions from different augmentations
        avg_preds = torch.mean(torch.stack(batch_preds), dim=0)
        _, predicted = torch.max(avg_preds, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Final accuracy with TTA: {final_accuracy:.2f} %')
print(f'Highest accuracy achieved during training: {best_acc:.2f} % (saved to {save_path})')