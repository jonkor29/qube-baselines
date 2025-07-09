import os
import yaml
import numpy as np

DEFAULT_CONFIG = "qube_servo3_usb"
def load_config(config_path=None):
    """
    Loads the configuration file 'config.yaml' containing the parameters for the selected Quanser Qube.

    The hardware configuration is selected by the environment variable QUANSER_HW. If the
    environment variable is not set, the default configuration DEFAULT_CONFIG is used.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    # Read the environment variable QUANSER_HW; default to 'qube_servo3_usb' if not set.
    selected_config = os.environ.get("QUANSER_HW", DEFAULT_CONFIG)
    if selected_config not in configs:
        raise ValueError(f"Configuration '{selected_config}' is not defined in {config_path}.")
    return configs[selected_config]

def params_from_config_dict(config_dict):
    """
    Converts the configuration dictionary to a list of parameters for the Qube.
    The parameters are in the order:
    [Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g]
    """
    params = []
    for key in ['Rm', 'kt', 'km', 'mr', 'Lr', 'Dr', 'mp', 'Lp', 'Dp', 'g']:
        params.append(config_dict[key])
    return np.array(params)