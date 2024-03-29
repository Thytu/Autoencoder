from tqdm import tqdm
from torch import no_grad


def train(dataloader, autoencoder, criterion, optimizer, device='cpu') -> float:
    train_loss = 0.0

    for (images, _) in tqdm(dataloader, leave=False):
        images = images.to(device)

        optimizer.zero_grad()

        pred, sigma, mu = autoencoder(images)

        loss = criterion(pred, images.view(images.size(0), 28 * 28), sigma, mu)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)
          
    train_loss = train_loss / len(dataloader)

    return train_loss


def test(dataloader, autoencoder, criterion, device='cpu') -> float:
    test_loss = 0.0

    with no_grad():
        for (images, _) in tqdm(dataloader, leave=False):
            images = images.to(device)

            pred, sigma, mu = autoencoder(images)

            loss = criterion(pred, images.view(images.size(0), 28 * 28), sigma, mu)

            test_loss += loss.item() * images.size(0)
            
        test_loss = test_loss / len(dataloader)

    return test_loss
