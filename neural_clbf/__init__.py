"""neural_clbf package initialization.

Includes compatibility stub for cvxpylayers which is referenced in saved
checkpoints but cannot be installed on modern systems.
"""
import sys
import types

# Stub for cvxpylayers (referenced in checkpoint but not needed for eval)
# This must be done before any torch.load() of checkpoints that were saved
# with cvxpylayers installed.
if 'cvxpylayers' not in sys.modules:
    class CvxpyLayer:
        """Stub class for checkpoint unpickling compatibility."""
        pass

    cvxpylayers = types.ModuleType('cvxpylayers')
    cvxpylayers_torch = types.ModuleType('cvxpylayers.torch')
    cvxpylayers_torch_cvxpylayer = types.ModuleType('cvxpylayers.torch.cvxpylayer')
    cvxpylayers_torch_cvxpylayer.CvxpyLayer = CvxpyLayer
    cvxpylayers.torch = cvxpylayers_torch
    cvxpylayers_torch.cvxpylayer = cvxpylayers_torch_cvxpylayer
    cvxpylayers_torch.CvxpyLayer = CvxpyLayer
    sys.modules['cvxpylayers'] = cvxpylayers
    sys.modules['cvxpylayers.torch'] = cvxpylayers_torch
    sys.modules['cvxpylayers.torch.cvxpylayer'] = cvxpylayers_torch_cvxpylayer
