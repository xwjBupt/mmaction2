from collections import OrderedDict, namedtuple
import torch
import re


def load_checkpoint(filename, revise_keys=[(r"^module\.", ""), (r"^backbone\.", "")]):
    """Load checkpoint from a file or URI.
    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename)  # map_location=torch.device('cpu')

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata
    checkpoint["state_dict"] = state_dict
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    torch.save(checkpoint, filename.replace(".pth", "#revised.pth"))


if __name__ == "__main__":
    load_checkpoint(
        "/ai/mnt/code/mmaction2/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb_20221021-f392cd2d.pth"
    )
    print("revised")
