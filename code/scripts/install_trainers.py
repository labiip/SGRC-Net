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
    source_dir = Path(sgrc_net.__file__).parent / "nnunet"
    destination_dir = Path(inspect.getfile(trainer_module)).parent
    names = ("nnUNetTrainerSGRCNet.py", "nnUNetTrainerSGRCNetCoW.py")
    for name in names:
        source = source_dir / name
        destination = destination_dir / name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(destination)
    for name in names:
        shutil.copy2(source_dir / name, destination_dir / name)
        print(f"Installed {name}")


if __name__ == "__main__":
    main()
