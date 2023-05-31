from ..classes import Face, GlobalState, From, To, As
from typing import Generator, Union
import logging

logger = logging.getLogger(__name__)


class FaceProcessor:
    def __init__(self, globalState: GlobalState):
        self.__state = globalState

    def gen_faces(self) -> Generator[Face, None, None]:
        for face in self.__state.faces:
            yield face

    def convert_labelType(self, value: Union[str, int], to: str) -> any:
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
            for name, number in self.__state.label.items():
                if number == value:
                    return name
        elif to == "Number":
            return self.__state.label.get(value, -1)
        return None

    def set_label(self, person: Union[list, dict]) -> None:
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
            self.__state.label = person
        elif type(person) == list:
            for i in range(len(person)):
                self.__state.label[person[i]] = i

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
        assert isinstance(
            target, str), "target must be 'From.LABEL' or 'From.FILENAME'."
        assert isinstance(value, str), "value must be a string."

        for face in self.gen_faces():
            if target == "Label":
                if face.label == value:
                    return face
            elif target == "Filename":
                if value in face.filenames:
                    return face
        return None
    
    def get_faces(self) -> list[Face]:
        return self.__state.faces
