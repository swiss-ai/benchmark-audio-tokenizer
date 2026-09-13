import importlib.metadata as metadata
import json
import sys

import torch

print(json.dumps({
    "python": sys.version,
    "distributions": sorted((d.metadata.get("Name"), d.version, str(d._path)) for d in metadata.distributions()),
    "torch": torch.__version__, "torch_file": torch.__file__,
    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
}, indent=2))
