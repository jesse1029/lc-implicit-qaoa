import sys

import numpy
import scipy

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)

try:
    import cupy as cp

    print("cupy", cp.__version__)
    x = cp.arange(8)
    print("cupy_sum", int(x.sum().get()))
    print("cupy_devices", cp.cuda.runtime.getDeviceCount())
except Exception as exc:
    print("cupy_error", repr(exc))
    raise
