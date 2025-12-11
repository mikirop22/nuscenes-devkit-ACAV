def plot_loss_curve(loss_history):
    """
    Plot the training loss curve.

    Args:
        loss_history (list or array): Sequence of loss values recorded during training.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid(True)
    plt.show()

import torch
import os

def save_model(model, optimizer, epoch, filename="mtp_model.pth"):
    """
    Guarda el modelo en python-sdk/main/models/{filename}
    """
    # Ruta absoluta relativa al proyecto
    base_dir = os.path.dirname(os.path.dirname(__file__))  # -> python-sdk
    models_dir = os.path.join(base_dir, "main", "models")

    # Crear carpeta si no existe
    os.makedirs(models_dir, exist_ok=True)

    save_path = os.path.join(models_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint, save_path)

    print(f"Model saved in: {save_path}")

