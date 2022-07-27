# Train.py -- sub-program to train the neural network.

from CommandLine.Program import Program
import argparse
import pathlib


class Train(Program):
    def Name(self):
        return "train"

    def Description(self):
        return "Train the neural network from images and manual segmentations."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath",
                            help="Path to image and segmentation data. "
                                 "Directory with subfolders training/ and validation/ with "
                                 "respective subfolders images/ and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("saveDirectory", type=pathlib.Path,
                            help="Directory where the trained model will be saved.")
        parser.add_argument("saveName", type=str,
                            help="Model name (will be used as prefix for saved models.)")
        parser.add_argument("-B", dest='batchSize', default=8,
                            help="Number of images to use for each backpropagation pass.", type=int)
        parser.add_argument("-M", dest='modelPath', default=None,
                            help="Path to existing model to continue training for",
                            type=pathlib.Path)
        parser.add_argument("-E", dest='epochs', default=100,
                            help="How many times the full image dataset should be learned.",
                            type=int)
        parser.add_argument("-DR", dest='dropoutRate', default=0.2,
                            help="Dropout rate of CNN during training.", type=float)
        parser.add_argument("-LR", dest='learningRate', default=0.001,
                            help="Neural network learning rate.", type=float)
        parser.add_argument("-P", dest='patience', default=5,
                            help="Stop training after this many epochs with no improvement.",
                            type=int)
        parser.add_argument("-S", dest='size', nargs=2, default=[512, 512],
                            help="Size of input images (e.g. -S 512 512). Not used if -M is "
                                 "specified.", type=int)
        parser.add_argument("--lite", action="store_true",
                            help="If set, the model will be saved as a TFLite model.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from Core.Model import TrainModel, BuildModel, GroundTruth, LoadModel

        if parserArgs.modelPath is not None:
            model = LoadModel(parserArgs.modelPath)
        else:
            model = BuildModel(parserArgs.size, parserArgs.dropoutRate, 8)

        self.MakeDirectory(parserArgs.saveDirectory)

        trainingPath: pathlib.Path = parserArgs.inputPath / "training"
        validationPath: pathlib.Path = parserArgs.inputPath / "validation"
        trainingImagesPath = trainingPath / "images"
        trainingSegmentationsPath = trainingPath / "segmentations"
        validationImagesPath = validationPath / "images"
        validationSegmentationsPath = validationPath / "segmentations"
        [self.AssertDirectoryExists(x) for x in [trainingPath, validationPath,
                                                 trainingImagesPath, trainingSegmentationsPath,
                                                 validationImagesPath, validationSegmentationsPath]]

        trainingData = [GroundTruth(x, y) for x, y in
                        zip(trainingImagesPath.iterdir(), trainingSegmentationsPath.iterdir())]
        validationData = [GroundTruth(x, y) for x, y in
                          zip(validationImagesPath.iterdir(),
                              validationSegmentationsPath.iterdir())]
        TrainModel(model, parserArgs.learningRate, parserArgs.patience, parserArgs.epochs,
                   parserArgs.batchSize, trainingData, validationData, parserArgs.saveDirectory,
                   parserArgs.saveName, parserArgs.lite)
