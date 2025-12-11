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
