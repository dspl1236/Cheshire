# Cheshire: Meshroom 2023.3's DepthMap node with a larger block.
#
# Installed by the pairing scripts next to Meshroom's compiled DepthMap.pyc; Python prefers the
# .py, so this module is what Meshroom loads. It loads the original compiled node and re-declares
# the class with its parallelization changed. Nothing else differs: the same attributes, the same
# command line, the same UID, so an existing cache stays valid.
#
# Why: each chunk is a separate aliceVision_depthMapEstimation process that loads the SfM data,
# probes the device and starts with a cold image cache - 19-27 s before its first tile on the
# 884-view set, plus a cold first batch - and Meshroom's block of 12 views makes 74 of them. With
# a block of 48 there are 19. (docs/04, 0.3.4 "the depth-map node was loading images".)
# CHESHIRE_DEPTHMAP_BLOCK sets another size; 0 restores Meshroom's 12.
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("meshroom.nodes.aliceVision._DepthMapBuiltin", os.path.join(_here, "DepthMap.pyc"))
_builtin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builtin)

_block = int(os.environ.get("CHESHIRE_DEPTHMAP_BLOCK", "48") or "0") or 12


class DepthMap(_builtin.DepthMap):
    parallelization = _builtin.desc.Parallelization(blockSize=_block)
