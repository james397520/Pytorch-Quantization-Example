import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST



if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    torch.float16

    test_dataset = MNIST('data', train=False, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    quantized_model_path = 'model/4bit/checkpoint_quantized.pt'
    
    # load TorchScript model
    print("Load TorchScript model...")
    model = torch.jit.load(quantized_model_path, map_location='cpu')


    print("Test quantized model...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the quantized model on the test images: {:.2f}%'.format(
        100 * correct / total))
