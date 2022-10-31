from tqdm import tqdm
from torch import no_grad


def train(dataloader, autoencoder, criterion, optimizer, device='cpu') -> float:
    train_loss = 0.0

    for (images, _) in tqdm(dataloader, leave=False):
        images = images.to(device)

        optimizer.zero_grad()

        outputs = autoencoder(images)

        loss = criterion(outputs, images)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
          
    train_loss = train_loss / len(dataloader)

    return train_loss


def test(dataloader, autoencoder, criterion, device='cpu') -> float:
    test_loss = 0.0

    with no_grad():
        for (images, _) in tqdm(dataloader, leave=False):
            images = images.to(device)

            outputs = autoencoder(images)

            loss = criterion(outputs, images)

            test_loss += loss.item()
            
        test_loss = test_loss / len(dataloader)

    return test_loss
