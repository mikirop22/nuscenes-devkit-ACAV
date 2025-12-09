# %% [markdown]
# # Entrenamiento y prueba de MTP con datos reales de nuScenes

# %%
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

# Modelos
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss

# %% [markdown]
# ## 1. Cargar dataset y helper
# Ajusta DATAROOT a tu ruta de nuScenes

# %%
DATAROOT = '/data/sets/nuscenes'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
helper = PredictHelper(nuscenes)

# %% [markdown]
# ## 2. Definir splits de entrenamiento y validación

# %%
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
train_split = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
val_split = get_prediction_challenge_split("mini_val", dataroot=DATAROOT)

print("Primeros 5 agentes del split de entrenamiento:", train_split[:5])

# %% [markdown]
# ## 3. Input representation

# %%
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

# %% [markdown]
# ## 4. Dataset real para PyTorch

# %%
class NuScenesDataset(Dataset):
    def __init__(self, split, helper, input_representation, device='cpu'):
        self.split = split
        self.helper = helper
        self.input_representation = input_representation
        self.device = device

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        instance_token, sample_token = self.split[idx].split("_")
        # Imagen de entrada (rasterizada)
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img_tensor = torch.Tensor(img).permute(2,0,1)  # C,H,W

        # Vector de estado del agente
        agent_state = torch.Tensor([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                    self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                    self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

        # Trajectory ground truth (futuro 3s)
        future_xy = self.helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
        gt = torch.Tensor(future_xy).unsqueeze(0)  # 1 x T x 2

        return img_tensor.to(self.device), agent_state.to(self.device), gt.to(self.device)

# %% [markdown]
# ## 5. Configuración de entrenamiento

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_modes = 2
batch_size = 4  # ajusta según GPU
dataset_train = NuScenesDataset(train_split, helper, input_representation, device=device)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Modelo
backbone = ResNetBackbone('resnet18')
model = MTP(backbone, num_modes=num_modes, n_hidden_layers=4096, input_shape=(3,100,100)).to(device)

# Pérdida y optimizador
loss_fn = MTPLoss(num_modes=num_modes, regression_loss_weight=1., angle_threshold_degrees=5.)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ## 6. Entrenamiento (ejemplo rápido)

# %%
loss_history = []
for n_iter, (img, agent, gt) in enumerate(dataloader_train):
    optimizer.zero_grad()
    pred = model(img, agent)
    loss = loss_fn(pred, gt)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if n_iter % 10 == 0:
        print(f"Iter {n_iter}, Loss: {loss.item():.4f}")

    if n_iter > 50:  # corto para ejemplo
        break

# %% [markdown]
# ## 7. Visualización de pérdida

# %%
plt.plot(loss_history)
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.title("Histórico de pérdida durante entrenamiento")
plt.show()

# %% [markdown]
# ## 8. Predicción de ejemplo

# %%
model.eval()
with torch.no_grad():
    img, agent, gt = dataset_train[0]
    pred = model(img.unsqueeze(0), agent.unsqueeze(0))  # batch=1
    print("Predicción shape:", pred.shape)
    print("Primeros 10 valores del primer modo:\n", pred[0,:10])
