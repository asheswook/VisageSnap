import face_recognition
from dataclasses import dataclass
import os
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.exceptions import NotFittedError
import pickle
from .classes import *
from .utils import *


# Make a class to semi-supervised the face recognition
class Core():
    def __init__(self):
        """
        VisageSnap Core Class
        ---------------------
        """
        _default_dir = os.getcwd()
        self.faces: list[Face] = []

        # Directory
        self.unlabeled_dir = os.path.join(_default_dir, "unlabeled")
        self.labeled_dir = os.path.join(_default_dir, "labeled")

        self.model_dir = os.path.join(_default_dir, "model", "face_model.pkl")

        self.predict_dir = os.path.join(_default_dir, "predict")

        self.label: dict = {}

        self.threshold = 0.42

        self.model = self._load_model()

    @staticmethod
    def _isImage(filename: str) -> bool:
        """
        This function checks if the file is an image file.
        
        Parameters
        ----------
        filename (str) : target filename.
        """
        assert isinstance(filename, str), "filename must be a string."
        
        list = [
            ".jpg",
            ".png",
            ".jpeg"
            ]

        for i in list:
            if filename.endswith(i):
                return True
        return False

    def get_faceObject(self, target: str, value: str) -> Face:
        """
        This function returns the face object with the given label.
        
        Parameters
        ----------
        target:
            - "From.LABEL" : label of the face object. (name of the person)
            - "From.FILENAME" : filename of the face object.
        
        value (str) : value of the target.
        """
        assert isinstance(target, str), "target must be 'From.LABEL' or 'From.FILENAME'."
        assert isinstance(value, str), "value must be a string."

        for face in self.gen_faces():
            if target == "Label":
                if face.label == value:
                    return face
            elif target == "Filename":
                if value in face.filenames:
                    return face
        return None

    def gen_faces(self) -> list[Face]:
        """
        This function returns the face list.
        """
        result = []
        for face in self.faces:
            yield face


    def _load_labeled(self) -> None: # 미리 주어지는 데이터는 한 사진에 한 사람만 있어야 한다.
        """
        This function loads the labeled data from the labeled directory.
        """
        for filename in gen(os.listdir(self.labeled_dir)):
            if self._isImage(filename):
                print("Loading labeled data: {}".format(filename))
                label = (filename.split(".")[0]).split("-")[0] # 파일 형식은 이름-번호.jpg
                image = face_recognition.load_image_file(os.path.join(self.labeled_dir, filename))
                encodings = face_recognition.face_encodings(image)[0]

                # 만약 두개의 얼굴이 같은 사진에 있다면
                if len(face_recognition.face_encodings(image)) > 1:
                    print("There are more than one face in the image: {}".format(filename))
                    continue
                
                # 같은 이름이 있는지 확인
                face_found = False
                for i, face in enumerate(self.gen_faces()):
                    if face.label == label:
                        print("The label is already in the list: {}".format(filename))
                        # 동일한 인코딩이 있는지 확인
                        if np.array_equal(face.encodings, encodings):
                            print("The encoding is already in the list: {}".format(filename))
                            continue
                        self.faces[i].encodings = np.vstack((face.encodings, encodings))
                        self.faces[i].filenames.append(filename)
                        face_found = True
                        break
                
                if not face_found:
                    self.faces.append(Face(label, encodings, [filename]))


    def _load_unlabeled(self) -> None:
        """
        This function loads the unlabeled data from the unlabeled directory.
        """
        for filename in gen(os.listdir(self.unlabeled_dir)):
            if self._isImage(filename):
                print("Loading unlabeled data: {}".format(filename))
                image = face_recognition.load_image_file(os.path.join(self.unlabeled_dir, filename))
                encodings = face_recognition.face_encodings(image)

                if len(encodings) == 0:
                    print("There is no face in the image: {}".format(filename))
                    continue

                for encoding in encodings:
                    self.faces.append(Face("unknown", encoding, [filename]))


    def _load_model(self) -> LabelPropagation:
        try:
            with open(self.model_dir, "rb") as f:
                self.model, self.faces = pickle.load(f)
                print("Model loaded.")
                return self.model
        except:
            print("There is no model in the model directory. create a new model.")
            self.model = LabelPropagation()
            self.faces = [] # 초기화
            return self.model

    def _save_model(self) -> None:
        data = (self.model, self.faces)
        with open(self.model_dir, "wb") as f:
            pickle.dump(data, f)

    def convert_labelType(self, value, to: str) -> any:
        """
        This function converts the label type. (numberLabel -> nameLabel, nameLabel -> numberLabel)
        
        Parameters
        ----------
        value (str or int) : target value.
        to (str) :
            - "To.NAME" : convert to name label.
            - "To.NUMBER" : convert to number label.

        Returns
        -------
        str or int : converted value. (if To.NAME, return str, if To.NUMBER, return int)
        """
        if to == "Name":
            for name, number in self.label.items():
                if number == value:
                    return name
        elif to == "Number":
            return self.label.get(value, -1)
        return None

    def set_label(self, person: any) -> None:
        """
        This function sets the label dictionary.
        
        Parameters
        ----------
        person (list or dict) : label list or dictionary.

        Example
        -------
        person = ["name1", "name2", "name3", ...]

        OR

        person = {
            "name1": 0,
            "name2": 1,
            "name3": 2,
            ...
        }

        - name1, name2, name3, ... : name of the person
        - 0, 1, 2, ... : number label (MUST NOT BE -1)
        """
        
        if type(person) == dict:
            self.label = person
        elif type(person) == list:
            for i in range(len(person)):
                self.label[person[i]] = i

    def set_directory(self, dicto: dict) -> None:
        """
        This function sets the directory.
        
        Parameters
        ----------
        dict (dict) : directory dictionary.

        Example
        -------
        dict = {
            "labeled": "labeled",
            "unlabeled": "unlabeled",
            "model": "model",
            "predict": "predict"
        }
        - labeled : directory of the labeled data
        - unlabeled : directory of the unlabeled data
        - model : directory of the model
        - predict : directory of the predict data

        Default
        -------
        labeled : "labeled"
        unlabeled : "unlabeled"
        model : "model"
        predict : "predict"
        """
        assert isinstance(dicto, dict), "parameter must be dictionary."

        if "labeled" in dicto:
            self.labeled_dir = dicto["labeled"]
        if "unlabeled" in dicto:
            self.unlabeled_dir = dicto["unlabeled"]
        if "model" in dicto:
            self.model_dir = dicto["model"]
        if "predict" in dicto:
            self.predict_dir = dicto["predict"]
        

    def _train(self, labeled: bool) -> None:
        assert isinstance(labeled, bool), "parameter must be boolean."
        
        if labeled:
            self._load_labeled()
        else:
            self._load_unlabeled()

        t_names = []
        t_encodings =[]

        for face in self.gen_faces():
            for encoding in gen(face.encodings):
                numberLabel = self.convert_labelType(face.label, To.NUMBER)
                if labeled and numberLabel == -1: # 라벨링 데이터 학습인데 unknown이면 학습하지 않음
                    continue
                t_names.append(numberLabel)
                t_encodings.append(encoding)

        t_encodings = np.array(t_encodings)
        t_names = np.array(t_names)
        
        print("Training the labeled data...")
        self.model.fit(t_encodings, t_names)
        self._save_model()
        print("Labeled training is done. The model is saved in the model directory.")

    def train_labeled_data(self) -> None:
        self._train(As.LABELED)

    def train_unlabeled_data(self) -> None:
        self._train(As.UNLABELED)

    
    @staticmethod
    def _get_average(face: Face) -> np.array:
        """
        This function returns the average of the encodings.
        
        Parameters
        ----------
        face (Face) : target face.
        """
        assert isinstance(face, Face), "parameter must be Face class."
        return np.average(face.encodings, axis=0)

    @staticmethod
    def _get_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        This function returns the distance between two encodings.
        
        Parameters
        ----------
        encoding1 (np.array) : encoding1.
        encoding2 (np.array) : encoding2.
        """
        assert isinstance(encoding1, np.ndarray), "parameter must be numpy array."
        assert isinstance(encoding2, np.ndarray), "parameter must be numpy array."

        return np.linalg.norm(encoding1 - encoding2)

    def _isNotUnknown(self, encoding) -> bool:
        """
        This function checks whether the encoding is unknown.
        
        Parameters
        ----------
        encoding (np.array) : target encoding.
        """
        assert isinstance(encoding, np.ndarray), "parameter must be numpy array."
        print("Checking whether the encoding is unknown...")
        min_distance = 1
        for face in self.gen_faces():
            print(face.label)
            average = self._get_average(face) # 저장된 얼굴 평균 구하고
            distance = self._get_distance(encoding, average) # 타겟과의 거리를 구한다
            if distance < min_distance:
                min_distance = distance

        print("min_distance : ", min_distance)
        print("ended")

        if min_distance < self.threshold:
            print("아는사람")
            return True # 모르는 사람이 아니다
        print("모르는사람")
        return False # 모르는 사람이다

    def predict(self, image: np.ndarray) -> list:
        assert isinstance(image, np.ndarray), "parameter must be numpy array."

        target_encodings = face_recognition.face_encodings(image)
        if len(target_encodings) == 0:
            return None
        
        result = []
        for target_encoding in target_encodings:
            if self._isNotUnknown(target_encoding): # 모르는 사람이 아니면
                result.append(self.model.predict([target_encoding])[0])
            else:
                result.append(-1)
        
        return result



    def predict_all(self) -> dict:
        result = {}
        for filename in os.listdir(self.predict_dir):
            if self._isImage(filename) == False:
                return
            
            image = face_recognition.load_image_file(os.path.join(self.predict_dir, filename))
            prediction = self.predict(image)

            if prediction == None:
                raise("There is no face in the image.")
            
            if len(prediction) == 1:
                result[filename] = self.convert_labelType(prediction[0], To.NAME)
            else:
                result[filename] = []
                for p in prediction:
                    result[filename].append(self.convert_labelType(p, To.NAME))
        return result