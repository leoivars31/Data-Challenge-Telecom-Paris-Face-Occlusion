import numpy as np

convnextv2_base = {
    0: ["data/submissions/checkpoints/kfold_20260610_025055", 0.001822],
    1: ["data/submissions/checkpoints/kfold_20260610_031335", 0.001603],
    2: ["data/submissions/checkpoints/kfold_20260610_044757", 0.001189],
    3: ["data/submissions/checkpoints/kfold_20260610_062159", 0.001365],
    4: ["data/submissions/checkpoints/kfold_20260610_063613", 0.001340],
}

mean_convnextv2_base = np.mean([v[1] for v in convnextv2_base.values()])
std_convnextv2_base = np.std([v[1] for v in convnextv2_base.values()])

eva02_base = {
    0: ["data/submissions/checkpoints/kfold_20260610_064628", 0.001686],
    1: ["data/submissions/checkpoints/kfold_20260610_073712", 0.001839],
    2: ["data/submissions/checkpoints/kfold_20260610_091219", 0.001245],
    3: ["data/submissions/checkpoints/kfold_20260610_112136", 0.001527],
    4: ["data/submissions/checkpoints/kfold_20260610_112454", 0.001362],
}

mean_eva02_base = np.mean([v[1] for v in eva02_base.values()])
std_eva02_base = np.std([v[1] for v in eva02_base.values()])

convnext_tiny = {
    0: ["data/submissions/checkpoints/kfold_20260610_124017", 0.001761],
    1: ["data/submissions/checkpoints/kfold_20260610_133205", 0.001714],
    2: ["data/submissions/checkpoints/kfold_20260610_150730", 0.001127],
    3: ["data/submissions/checkpoints/kfold_20260610_152706", 0.001479],
    4: ["data/submissions/checkpoints/kfold_20260610_153347", 0.001449],
}

mean_convnext_tiny = np.mean([v[1] for v in convnext_tiny.values()])
std_convnext_tiny = np.std([v[1] for v in convnext_tiny.values()])

print("ConvNeXtV2 Base:", convnextv2_base)
print("Mean ConvNeXtV2 Base:", mean_convnextv2_base)
print("Std ConvNeXtV2 Base:", std_convnextv2_base)
print("Eva02 Base:", eva02_base)
print("Mean Eva02 Base:", mean_eva02_base)
print("Std Eva02 Base:", std_eva02_base)
print("ConvNeXt Tiny:", convnext_tiny)
print("Mean ConvNeXt Tiny:", mean_convnext_tiny)
print("Std ConvNeXt Tiny:", std_convnext_tiny)

"""
Premier des 3 sans early stopping (846 274-6); Le suivant sans early stopping + random erasing (846 820-2)
"""

import torch
print(torch.__version__)