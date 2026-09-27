# Cheshire: Meshroom 2025.1's DepthMap node with a larger block.
#
# Installed by the pairing scripts as <Meshroom>/aliceVision/share/meshroom/aliceVision/DepthMap.py,
# with Meshroom's own file kept beside it as DepthMap.py.meshroom, a name Meshroom's node loader does
# not import. This module loads that file and re-declares the class with its parallelization
# changed. Nothing else differs: the same attributes, the same version, the same command line, the
# same UID, so an existing cache stays valid.
#
# Meshroom 2025.1 ships its node descriptions as source in the AliceVision tree, where 2023.3
# compiled them into lib/meshroom/nodes, so the 2023.3 override (DepthMap.py beside this file, which
# loads the .pyc next to it) has nothing to load there. Why the larger block: see that file.
# CHESHIRE_DEPTHMAP_BLOCK sets another size; 0 restores Meshroom's 12.
import importlib.machinery
import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DepthMap.py.meshroom")
_loader = importlib.machinery.SourceFileLoader(__name__ + "_meshroom", _path)
_builtin = importlib.util.module_from_spec(importlib.util.spec_from_loader(_loader.name, _loader))
_loader.exec_module(_builtin)

__version__ = _builtin.__version__
_block = int(os.environ.get("CHESHIRE_DEPTHMAP_BLOCK", "48") or "0") or 12


class DepthMap(_builtin.DepthMap):
    parallelization = _builtin.desc.Parallelization(blockSize=_block)
