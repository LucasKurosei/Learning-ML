import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# environment variables
EPOCHS = 5
BATCH_SIZE = 64
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

'''
if __name__ == "__main__":
    train_loader = DataLoader(training_data, BATCH_SIZE, True)
    test_loader = DataLoader(test_data, BATCH_SIZE, True)
    model = NeuralNetwork().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")
'''