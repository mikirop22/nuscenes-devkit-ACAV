import json
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.compute_metrics import compute_metrics

# src/evaluate.py

import json
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.compute_metrics import compute_metrics

def load_model(model, optimizer, filename="mtp_model.pth", device="cpu"):
    """
    Carga un modelo desde python-sdk/main/models/{filename}
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # -> python-sdk
    models_dir = os.path.join(base_dir, "main", "models")
    load_path = os.path.join(models_dir, filename)

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", None)

    print(f"📂 Modelo cargado desde: {load_path}")
    print(f"🔄 Última epoch entrenada: {epoch}")

    return model, optimizer, epoch


def evaluate_submission(version, dataroot, submission_path,
                        config_name="predict_2020_icra.json",
                        verbose=True):
    """
    Evaluate a submission.json file using the official nuScenes metrics.

    Args:
        version (str): nuScenes version, e.g. "v1.0-mini"
        dataroot (str): Path to the nuScenes dataset
        submission_path (str): Path to submission.json file
        config_name (str): Metric configuration file
        verbose (bool): If True, prints progress and displays results

    Returns:
        dict: metrics dictionary returned by the nuScenes evaluator
    """
    if verbose:
        print("Loading nuScenes and metric configuration...")

    nusc = NuScenes(version=version, dataroot=dataroot)
    helper = PredictHelper(nusc)
    config = load_prediction_config(helper, config_name)

    if verbose:
        print("Loading predictions from submission file...")

    predictions = json.load(open(submission_path, "r"))

    if verbose:
        print("Computing metrics...")

    results = compute_metrics(predictions, helper, config)

    output_metrics = submission_path.replace(".json", "_metrics.json")
    json.dump(results, open(output_metrics, "w"), indent=2)

    if verbose:
        print(f"Metrics saved to {output_metrics}\n")
        try:
            from IPython.display import display
            display(results)
        except:
            print(results)

    return results
