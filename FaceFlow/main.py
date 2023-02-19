import face_recognition
from dataclasses import dataclass
import os
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.exceptions import NotFittedError
import pickle

@dataclass
class Face():
    label: str
    encodings: np.ndarray
    filenames: list

@dataclass
class From():
    LABEL = "Label"
    FILENAME = "Filename"

@dataclass
class To():
    NAME = "Name"
    NUMBER = "Number"

# Make a class to semi-supervised the face recognition
class FaceCore():
    def __init__(self):
        """
        FaceCore
        --------
        """
        _default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.faces = []

        # Directory
        self.unlabeled_dir = os.path.join(_default_dir, "unlabeled")
        self.labeled_dir = os.path.join(_default_dir, "labeled")

        self.model_dir = os.path.join(_default_dir, "model", "face_model.pkl")

        self.predict_dir = os.path.join(_default_dir, "predict")

        self.label: dict = {}

    @staticmethod
    def _isImage(filename: str) -> bool:
        """
        This function checks if the file is an image file.
        
        Parameters
        ----------
        filename (str) : target filename.
        """
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
        target (str) :
            - "From.LABEL" : label of the face object. (name of the person)
            - "From.FILENAME" : filename of the face object.
        
        value (str) : value of the target.
        """
        for face in self.faces:
            if target == "Label":
                if face.label == value:
                    return face
            elif target == "Filename":
                if value in face.filenames:
                    return face
        return None



    def _load_labeled(self): # 미리 주어지는 데이터는 한 사진에 한 사람만 있어야 한다.
        """
        This function loads the labeled data from the labeled directory.
        """
        for filename in os.listdir(self.labeled_dir):
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
                for face in self.faces:
                    if face.label == label:
                        print("The label is already in the list: {}".format(filename))
                        # 동일한 인코딩이 있는지 확인
                        if np.array_equal(face.encodings, encodings):
                            print("The encoding is already in the list: {}".format(filename))
                            continue
                        face.encodings = np.vstack((face.encodings, encodings))
                        face.filenames.append(filename)
                        face_found = True
                        break
                
                if not face_found:
                    self.faces.append(Face(label, encodings, [filename]))


    def _load_unlabeled(self):
        """
        This function loads the unlabeled data from the unlabeled directory.
        """
        for filename in os.listdir(self.unlabeled_dir):
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
                model, self.faces = pickle.load(f)
                print("Model loaded.")
                return model
        except:
            print("There is no model in the model directory. create a new model.")
            model = LabelPropagation()
            self.faces = [] # 초기화
            return model

    def _save_model(self, model):
        with open(self.model_dir, "wb") as f:
            pickle.dump((model, self.faces), f)

    def convert_labelType(self, value, to: str):
        """
        This function converts the label type. (numberLabel -> nameLabel, nameLabel -> numberLabel)
        
        Parameters
        ----------
        value (str or int) : target value.
        to (str) :
            - "To.NAME" : convert to name label.
            - "To.NUMBER" : convert to number label.
        """
        if to == "Name":
            for name, number in self.label.items():
                if number == value:
                    return name
        elif to == "Number":
            return self.label.get(value, -1)
        return None

    def set_label(self, person):
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

    def set_directory(self, dict: dict):
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
            "model": "model"
        }
        - labeled : directory of the labeled data
        - unlabeled : directory of the unlabeled data
        - model : directory of the model

        Default
        -------
        labeled : "labeled"
        unlabeled : "unlabeled"
        model : "model"
        """
        if "labeled" in dict:
            self.labeled_dir = dict["labeled"]
        if "unlabeled" in dict:
            self.unlabeled_dir = dict["unlabeled"]
        if "model" in dict:
            self.model_dir = dict["model"]
        

    def _train(self, labeled: bool):
        if labeled:
            self._load_labeled()
        else:
            self._load_unlabeled()

        t_names = []
        t_encodings =[]

        for face in self.faces:
            for encoding in face.encodings:
                numberLabel = self.convert_labelType(face.label, To.NUMBER)
                if labeled and numberLabel == -1:
                    continue
                t_names.append(numberLabel)
                t_encodings.append(encoding)

        t_encodings = np.array(t_encodings)
        t_names = np.array(t_names)
        
        model = self._load_model()
        print("Training the labeled data...")
        model.fit(t_encodings, t_names)
        self._save_model(model)
        print("Labeled training is done. The model is saved in the model directory.")

    def train_labeled_data(self):
        self._train(True)

    def train_unlabeled_data(self):
        self._train(False)

    def predict(self, image):
        try:
            model = self._load_model()
            encodings = face_recognition.face_encodings(image)
            names = model.predict(encodings)
            return names
        except NotFittedError:
            print("The model is not trained yet. Train the model first.")
            return None
        

    def predict_all(self):
        result = {}
        for filename in os.listdir(self.predict_dir):
            if self._isImage(filename):
                image = face_recognition.load_image_file(os.path.join(self.predict_dir, filename))
                numberLabel = self.predict(image)
                nameLabel = self.convert_labelType(numberLabel, To.NAME)
                result[filename] = nameLabel
        return result
