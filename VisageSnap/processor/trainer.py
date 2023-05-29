from ..classes import Face, GlobalState, Directory, From, To, As
from .faceprocessor import FaceProcessor
from ..image import ImageLoader
from ..model import ModelHandler
import numpy as np

class Trainer(FaceProcessor):
    def __init__(self, globalState: GlobalState = None, directory: Directory = None, imageLoader: ImageLoader = None, modelHandler: ModelHandler = None):
        super().__init__(globalState)

        self.__state = globalState
        self.__directory = directory
        self.imageLoader = ImageLoader(self.__state, self.__directory) if imageLoader is None else imageLoader
        self.modelHandler = ModelHandler(self.__state, self.__directory) if modelHandler is None else modelHandler

    def __train(self, labeled: bool) -> None:
        assert isinstance(labeled, bool), "parameter must be boolean."

        if labeled:
            self.imageLoader.load_labeled()
        else:
            self.imageLoader.load_unlabeled()

        t_names = []
        t_encodings =[]

        for face in self.gen_faces():
            for encoding in face.encodings:
                numberLabel = self.convert_labelType(face.label, To.NUMBER)
                if labeled and numberLabel == -1: # 라벨링 데이터 학습인데 unknown이면 학습하지 않음
                    continue
                t_names.append(numberLabel)
                t_encodings.append(encoding)

        t_encodings = np.array(t_encodings)
        t_names = np.array(t_names)

        self.__state.model.fit(t_encodings, t_names)
        self.modelHandler.save()

    def train_labeled_data(self) -> None:
        self.__train(As.LABELED)

    def train_unlabeled_data(self) -> None:
        self.__train(As.UNLABELED)