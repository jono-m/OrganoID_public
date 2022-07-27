from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from PIL import Image
from Core.Model import PrepareImagesForModel, LoadLiteModel, Detect
from Core.ImageHandling import ImagesToHeatmaps, LabeledImagesToColoredImages, LoadPILImages
from Core.Identification import Label, Cleanup, DetectEdges

outputPath = Path(r"Publication\Figure2\2a\ExampleImages")
outputPath.mkdir(parents=True, exist_ok=True)

model = LoadLiteModel(Path(r"Publication\Model\model.tflite"))
images = PrepareImagesForModel(LoadPILImages(Path(r"Publication\Dataset\testing\Timelapse.tif")),
                               model)
prepared = images[26][None]
detected = Detect(model, prepared)
edges = DetectEdges(detected, 2, 0.005, 0.05, 0.5)
heatmap = ImagesToHeatmaps(detected)
labeled = LabeledImagesToColoredImages(Cleanup(Label(detected, edges, 0.5, 2), 100, True, True),
                                       colors=[(255, 198, 30),
                                               (175, 88, 186),
                                               (0, 205, 108),
                                               (0, 154, 222)])
Image.fromarray(prepared[0]).save(outputPath / "original.png")
Image.fromarray(heatmap[0]).save(outputPath / "detected.png")
Image.fromarray(edges[0]).save(outputPath / "edges.png")
Image.fromarray(labeled[0]).save(outputPath / "labeled.png")
