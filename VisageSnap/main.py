from .model import ModelHandler
from .image import ImageLoader
from .processor import FaceProcessor, Trainer, Predictor
from .classes import GlobalState, Directory
import numpy as np

class Core(FaceProcessor):
    """
    VisageSnap Core Class
    ---------------------
    """
    def __init__(self, globalState: GlobalState = None, directory: Directory = None):
        self.__state = GlobalState([], {}, None) if globalState is None else globalState
        self.__directory = Directory("labeled", "unlabeled", "model", "predict") if directory is None else directory

        super().__init__(self.__state)

        self.modelHandler = ModelHandler(self.__state, self.__directory)
        self.imageLoader = ImageLoader(self.__state, self.__directory)
        self.trainer = Trainer(self.__state, self.__directory, self.imageLoader, self.modelHandler)
        self.predictor = Predictor(self.__state, self.__directory)

    def set_directory(self, directory: Directory) -> None:
        self.__directory = directory

    @property
    def threshold(self) -> float:
        return self.predictor.threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        assert isinstance(value, float), "threshold must be float."
        self.predictor.threshold = value

    def load_model(self) -> None:
        """
        This function loads the model.
        If there is no model, it will create a new model.

        Also you can set the model directory by using set_directory method.
        """
        self.modelHandler.load()

    def is_model_loaded(self) -> bool:
        """
        This function checks whether the model is loaded or not.
        """
        return self.__state.model is not None

    def train_labeled_data(self) -> None:
        assert self.is_model_loaded(), "Model is not loaded."
        self.trainer.train_labeled_data()

    def train_unlabeled_data(self) -> None:
        assert self.is_model_loaded(), "Model is not loaded."
        self.trainer.train_unlabeled_data()

    def predict_encoding(self, encoding: np.ndarray) -> int:
        assert self.is_model_loaded(), "Model is not loaded."
        return self.predictor.predict_encoding(encoding)
    
    def predict_image(self, image: np.ndarray) -> int:
        assert self.is_model_loaded(), "Model is not loaded."
        return self.predictor.predict_image(image)
    
    def predict_all(self) -> dict:
        assert self.is_model_loaded(), "Model is not loaded."
        return self.predictor.predict_all()