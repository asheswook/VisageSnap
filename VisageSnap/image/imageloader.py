from ..classes import Face, GlobalState, Directory, From, To, As
from ..utils import isimage, gen
import numpy as np
import face_recognition
import os

class ImageLoader:
    def __init__(self, globalState: GlobalState = None, directory: Directory = None):
        self.__state = globalState
        self.__directory = directory

    def load_labeled(self) -> None:
        """
        This function loads the labeled data from the labeled directory.
        Only allows one face per image.
        """
        for filename in gen(os.listdir(self.__directory.labeled)):
            if isimage(filename):
                label = (filename.split(".")[0]).split("-")[0]  # 파일 형식은 이름-번호.jpg임. 이름만 추출
                image = face_recognition.load_image_file(os.path.join(self.__directory.labeled, filename))
                encodings = face_recognition.face_encodings(image)
                encoding = encodings[0]
    
                # If there are more than one face, skip.
                if len(encodings) > 1:
                    continue

                # Check if the face is already in the state.
                FACE_FOUND = False
                for i, face in enumerate(self.__state.faces):  # Search for the face in the state

                    if face.label == label:  # If the face label is already in the state
                        for old_encoding in face.encodings:

                            # Check if the encoding is already in the face.
                            if np.array_equal(old_encoding, encoding):
                                continue

                            # If the encoding is not in the face, append it.
                            self.__state.faces[i].encodings.append(encoding)
                            self.__state.faces[i].filenames.append(filename)
                            FACE_FOUND = True

                if not FACE_FOUND: # If the face is not in the state, create new face.
                    self.__state.faces.append(Face(label, [encoding], [filename]))

    def load_unlabeled(self) -> None:
        """
        This function loads the unlabeled data from the unlabeled directory.
        """
        for filename in gen(os.listdir(self.__directory.unlabeled)):
            if isimage(filename):
                image = face_recognition.load_image_file(os.path.join(self.__directory.unlabeled, filename))
                encodings = face_recognition.face_encodings(image)

                if len(encodings) == 0:  # If there is no face, skip.
                    continue

                for encoding in encodings:
                    self.__state.faces.append(Face("unknown", encoding, [filename]))
