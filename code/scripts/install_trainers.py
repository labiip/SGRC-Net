"""Install the public trainer names into an existing nnU-Net 2.5.2 environment."""
from pathlib import Path
import inspect
import shutil
from importlib.metadata import version
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
import sgrc_net


def main():
    if version("nnunetv2") != "2.5.2":
        raise RuntimeError("Use nnunetv2==2.5.2; this implementation uses its loader API.")
    source = Path(sgrc_net.__file__).parent / "nnunet" / "nnUNetTrainerSGRCNet.py"
    destination = Path(inspect.getfile(trainer_module)).parent / source.name
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    print(f"Installed {destination.name}")


if __name__ == "__main__":
    main()
