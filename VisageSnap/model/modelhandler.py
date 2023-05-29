from ..classes import Face, GlobalState, Directory, From, To, As
from sklearn.semi_supervised import LabelPropagation
import pickle
import os

class ModelHandler:
    def __init__(self, globalState: GlobalState = None, directory: Directory = None):
        self.__state = globalState
        self.__directory = directory
        self.__state.model = None

    def load(self) -> None:
        try:
            model_path = os.path.join(self.__directory.model, "face_model.pkl")

            with open(model_path, "rb") as f:
                self.__state.model, self.__state.faces = pickle.load(f)
                print("Model loaded.")

        except:
            print("Model not found. Creating new model...")
            self.__state.model = LabelPropagation()
            self.__state.faces = []  # 초기화
        
    def save(self) -> None:
        model_path = os.path.join(self.__directory.model, "face_model.pkl")

        # Create directory if not exists
        os.mkdir(self.__directory.model) if not os.path.exists(self.__directory.model) else None

        with open(model_path, "wb") as f:
            pickle.dump((self.__state.model, self.__state.faces), f)  # Save as tuple