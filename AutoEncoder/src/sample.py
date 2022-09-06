import torch
import numpy as np

from PIL import Image


def compare_original_with_reconstructed_images(autoencoder, dataloader):
    images, _ = iter(dataloader).next()

    output = autoencoder(images)[0]
    output = output.detach().numpy()
    reconstructed_image = np.moveaxis(output, 0, -1)

    original_image = images.numpy()[0]
    original_image = np.moveaxis(original_image, 0, -1)
    # image = np.swapaxes(image, 0, -1)

    img = Image.fromarray(original_image* 255, 'RGB')
    img.save('original.png')

    img = Image.fromarray(reconstructed_image* 255, 'RGB')
    img.save('reconstructed.png')

if __name__ == "__main__":
    from data_handler import get_testing_data

    # import matplotlib

    # matplotlib.use('nbagg')

    autoencoder = torch.load("autoencoder.pt").to(torch.device("cpu"))
    dataloader = get_testing_data()

    compare_original_with_reconstructed_images(autoencoder, dataloader)