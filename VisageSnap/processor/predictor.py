from ..classes import Face, GlobalState, Directory, From, To, As
from ..utils import isimage
from .faceprocessor import FaceProcessor
import numpy as np
import face_recognition
import os
from functools import cache


class Predictor(FaceProcessor):
    def __init__(self, globalState: GlobalState = None, directory: Directory = None):
        super().__init__(globalState)

        self.__state = globalState
        self.__directory = directory
        self.threshold = 0.48

    @staticmethod
    @cache
    def __get_average(face: Face) -> np.array:
        """
        This function returns the average of the encodings.

        Parameters
        ----------
        face (Face) : target face.
        """
        assert isinstance(face, Face), "parameter must be Face class."
        return np.average(face.encodings, axis=0)

    @staticmethod
    def __get_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        This function returns the distance between two encodings.

        Parameters
        ----------
        encoding1 (np.array) : encoding1.
        encoding2 (np.array) : encoding2.
        """
        assert isinstance(
            encoding1, np.ndarray), "parameter must be numpy array."
        assert isinstance(
            encoding2, np.ndarray), "parameter must be numpy array."

        return np.linalg.norm(encoding1 - encoding2)

    def __isNotUnknown(self, encoding) -> bool:
        """
        This function checks whether the encoding is unknown.

        Parameters
        ----------
        encoding (np.array) : target encoding.
        """
        assert isinstance(
            encoding, np.ndarray), "parameter must be numpy array."

        min_distance = 1
        for face in self.gen_faces():
            average = self.__get_average(face)  # 저장된 얼굴 평균 구하고
            distance = self.__get_distance(encoding, average)  # 타겟과의 거리를 구한다
            if distance < min_distance:
                min_distance = distance

        if min_distance < self.threshold:
            return True  # 모르는 사람이 아니다
        return False  # 모르는 사람이다

    def __predict(self, image: np.ndarray) -> list:
        assert isinstance(image, np.ndarray), "parameter must be numpy array."

        target_encodings = face_recognition.face_encodings(image)
        if len(target_encodings) == 0:
            return None

        result = []
        for target_encoding in target_encodings:
            if self.__isNotUnknown(target_encoding):  # 모르는 사람이 아니면
                result.append(self.__state.model.predict([target_encoding])[0])
            else:
                result.append(-1)

        return result

    def predict_image(self, image: np.ndarray) -> list:
        assert isinstance(image, np.ndarray), "parameter must be numpy array."

        return self.__predict(image)

    def predict_encoding(self, encoding: np.ndarray) -> int:
        assert isinstance(
            encoding, np.ndarray), "parameter must be numpy array."

        prediction = self.__predict([encoding])

        return -1 if -1 in prediction else prediction[0]

    def predict_all(self) -> dict:
        result = {}
        for filename in os.listdir(self.__directory.predict):
            if isimage(filename) == False:
                return

            image = face_recognition.load_image_file(
                os.path.join(self.__directory.predict, filename))
            prediction = self.__predict(image)

            if prediction == None:
                raise ("There is no face in the image.")

            if len(prediction) == 1:
                result[filename] = self.convert_labelType(
                    prediction[0], To.NAME)
            else:
                result[filename] = []
                for p in prediction:
                    result[filename].append(self.convert_labelType(p, To.NAME))
        return result
