from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadModel, ComputeIOUs
from Core.ImageHandling import LoadPILImages
import numpy as np

validationImages = LoadPILImages(Path(r"Publication\Dataset\validation\images"))
segmentationImages = LoadPILImages(Path(r"Publication\Dataset\validation\segmentations"))

modelPaths = Path(r"Publication\OrganoIDModel").iterdir()
for modelPath in modelPaths:
    model = LoadModel(modelPath)
    ious = ComputeIOUs(model, validationImages, segmentationImages)
    print(modelPath.name + ": " + str(np.mean(ious)) + " (SD: " + str(np.std(ious)) + ")")
