from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, Detect, PrepareImagesForModel
from Core.ImageHandling import ImagesToHeatmaps, LoadPILImages
from PIL import Image

model = LoadLiteModel(Path(r"Publication\Model\model.tflite"))
exampleImage = PrepareImagesForModel(
    LoadPILImages(Path(r"Publication\Dataset\training\original\images\50.png")), model)
detected = Detect(model, exampleImage)
heatmap = ImagesToHeatmaps(detected)
Image.fromarray(heatmap[0]).save(Path(r"Publication\Figure1\1b\ExampleImage.png"))
