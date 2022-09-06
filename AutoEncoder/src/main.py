from os import path
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from AutoEncoder import AutoEncoder

from training import train, test
from data_handler import get_testing_data, get_training_data

def main():
    EPOCHS = 100
    MODEL_OUTPUT_PATH = "autoencoder.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()

    autoencoder = AutoEncoder().to(device)
    # if path.exists(MODEL_OUTPUT_PATH):
    #     autoencoder.load_state_dict(torch.load(MODEL_OUTPUT_PATH))
    #     print("loading intermediate weights")
    # else:
    #     print("loading new weights")

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    train_loader = get_training_data(batch_size=32, pin_memory=True)
    test_loader = get_testing_data(batch_size=32, pin_memory=True)

    pbar = tqdm(range(EPOCHS))

    train_losses = []
    test_losses = []

    for _ in pbar:
        train_loss = train(train_loader, autoencoder, criterion, optimizer, device=device)
        test_loss = test(test_loader, autoencoder, criterion, device=device)

        pbar.set_description(f"loss: {train_loss:.2f} test loss:{test_loss:.2f}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    torch.save(autoencoder.state_dict(), MODEL_OUTPUT_PATH)

    plt.plot(range(EPOCHS), train_losses, 'g', label='Training loss')
    plt.plot(range(EPOCHS), test_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('evolution_of_the_losses.png')

if __name__ == "__main__":
    main()
