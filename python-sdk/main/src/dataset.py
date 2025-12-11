# ----------------------------------------
# Dataset y DataLoader para MTP
# ----------------------------------------
import torch
from torch.utils.data import Dataset

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
        
        # 1. Rasterized input representation
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img_tensor = torch.Tensor(img).permute(2,0,1)

        # 2. Agent dynamic state (values where nan)
        vel = self.helper.get_velocity_for_agent(instance_token, sample_token)
        acc = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        hcr = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        agent_state = torch.Tensor([vel, acc, hcr])
        
        # If there's a nan value --> 0.0 for agent that just appeared
        agent_state = torch.nan_to_num(agent_state, nan=0.0)

        # 3. Ground-truth future trajectory
        future_xy = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
        gt = torch.Tensor(future_xy).unsqueeze(0)

        return img_tensor.to(self.device), agent_state.to(self.device), gt.to(self.device)



class NuScenesMTPInferenceDataset(torch.utils.data.Dataset):
    """
    Dataset equivalente al de entrenamiento pero sin GT.
    Permite generar predicciones para cada instancia del split.
    """
    def __init__(self, split, helper, input_representation, device='cpu'):
        self.split = split
        self.helper = helper
        self.input_representation = input_representation
        self.device = device

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        instance_token, sample_token = self.split[idx].split("_")

        # Imagen rasterizada igual que en train
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img_tensor = torch.Tensor(img).permute(2,0,1).to(self.device)

        # Estado del agente (vel, acc, heading-change-rate)
        vel = self.helper.get_velocity_for_agent(instance_token, sample_token)
        acc = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        hcr = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        agent_state = torch.Tensor([vel, acc, hcr])
        agent_state = torch.nan_to_num(agent_state, nan=0.0).to(self.device)

        return img_tensor, agent_state, instance_token, sample_token