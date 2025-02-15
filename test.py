from gan import Generator
import torch
import matplotlib.pyplot as plt
import numpy as np

# Configuration
nz = 100  # Must match training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize generator
netG = Generator(ngpu=1).to(device)
netG.load_state_dict(torch.load('generator.pth'))
netG.eval()  # Set to evaluation mode

# Generate noise
noise = torch.randn(1, nz, 1, 1, device=device)

# Generate image
with torch.no_grad():
    fake = netG(noise).detach().cpu()

# Denormalize and plot
img = np.transpose(fake[0], (1, 2, 0))
img = (img * 0.5 + 0.5)  # Scale from [-1,1] to [0,1]
plt.axis('off')
plt.imshow(img)
plt.savefig('generated_image.png', bbox_inches='tight')
plt.show()