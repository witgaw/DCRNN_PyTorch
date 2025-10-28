"""DCRNN PyTorch - Diffusion Convolutional Recurrent Neural Network"""

from pathlib import Path

__version__ = "0.1.0"


def get_config_path(config_name):
    """
    Get the path to a bundled config file.

    Args:
        config_name: Name of the config file (e.g., 'dcrnn_bay.yaml', 'dcrnn_la.yaml')
                    or path relative to data/model/ (e.g., 'pretrained/PEMS-BAY/config.yaml')

    Returns:
        Path to the config file

    Examples:
        >>> get_config_path('dcrnn_bay.yaml')
        >>> get_config_path('pretrained/PEMS-BAY/config.yaml')
    """
    package_dir = Path(__file__).parent
    config_path = package_dir / 'data' / 'model' / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_name}")

    return str(config_path)
