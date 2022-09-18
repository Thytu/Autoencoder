import torch
import gradio as gr
import torchvision.transforms as transforms

from src.AutoEncoder import AutoEncoder

autoencoder = AutoEncoder(input_dim=28 * 28, hidden_dim=200, latent_dim=20).to('cpu').eval()
autoencoder.load_state_dict(torch.load("autoencoder.pt"))

def inference(*args):
    sigma = torch.tensor(args[:20], dtype=torch.float).unsqueeze(0)
    mu = torch.tensor(args[20:], dtype=torch.float).unsqueeze(0)

    z = autoencoder.create_latent_space(sigma, mu)
    generated_output = autoencoder.decode(z).view(1, 28, 28)

    return transforms.ToPILImage()(generated_output.cpu().clone().squeeze(0))


demo = gr.Interface(
    fn=inference,
    inputs=[gr.Slider(minimum=0, maximum=1, step=0.01, label=f"sigma_{idx + 1}" if idx < 20 else f"mu_{idx - 20 + 1}") for idx in range(20 * 2)],
    outputs=[gr.Image(shape=(28, 28), image_mode='L')],
    live=True,
)

demo.launch()