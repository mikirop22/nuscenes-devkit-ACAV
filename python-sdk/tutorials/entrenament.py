# ----------------------------------------
# Dataset y DataLoader para MTP
# ----------------------------------------
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss

class NuScenesMTPDataset(Dataset):
    def __init__(self, split, helper, input_representation, device='cpu'):
        self.split = split
        self.helper = helper
        self.input_representation = input_representation
        self.device = device

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        instance_token, sample_token = self.split[idx].split("_")
        
        # Imagen rasterizada
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img_tensor = torch.Tensor(img).permute(2,0,1)  # C,H,W

        # Vector de estado
        agent_state = torch.Tensor([
            self.helper.get_velocity_for_agent(instance_token, sample_token),
            self.helper.get_acceleration_for_agent(instance_token, sample_token),
            self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        ])

        # Trayectoria futura (ground truth)
        future_xy = self.helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
        gt = torch.Tensor(future_xy).unsqueeze(0)  # 1 x T x 2

        return img_tensor.to(self.device), agent_state.to(self.device), gt.to(self.device)


# ----------------------------------------
# Configuración de entrenamiento
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_modes = 2
batch_size = 4

dataset_train = NuScenesMTPDataset(mini_train, helper, mtp_input_representation, device=device)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Modelo
backbone = ResNetBackbone('resnet50')  # o 'resnet18'
model = MTP(backbone, num_modes=num_modes, n_hidden_layers=4096, input_shape=(3,100,100)).to(device)

# Pérdida y optimizador
loss_fn = MTPLoss(num_modes=num_modes, regression_loss_weight=1., angle_threshold_degrees=5.)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ----------------------------------------
# Bucle de entrenamiento simple
# ----------------------------------------
loss_history = []

model.train()
for n_iter, (img, agent, gt) in enumerate(dataloader_train):
    optimizer.zero_grad()
    pred = model(img, agent)
    loss = loss_fn(pred, gt)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if n_iter % 10 == 0:
        print(f"Iter {n_iter}, Loss: {loss.item():.4f}")

    if n_iter > 50:  # ejemplo rápido
        break


# ----------------------------------------
# Visualización de pérdida
# ----------------------------------------
plt.plot(loss_history)
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.title("Histórico de pérdida durante entrenamiento")
plt.show()


# ----------------------------------------
# Predicción de ejemplo
# ----------------------------------------
model.eval()
with torch.no_grad():
    img, agent, gt = dataset_train[0]
    pred = model(img.unsqueeze(0), agent.unsqueeze(0))  # batch=1
    print("Predicción shape:", pred.shape)
    print("Primeros 10 valores del primer modo:\n", pred[0,:10])

