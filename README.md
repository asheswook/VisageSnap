# VisageSnap

![Release](https://shields.io/github/v/release/asheswook/VisageSnap?display_name=tag&sort=semver) ![build](https://img.shields.io/github/actions/workflow/status/asheswook/VisageSnap/docker-workflow.yml?branch=main)

**English** | [한국어](README-Korean.md)

Recognizes faces and trains models, brings in the pictures and provides identification predictions and face classification. It also performs semi-supervised learning.

## Feature

- Recognize faces.
- Train the model through semi-supervised learning with labeled or unlabeled pictures.
- Predicts if the face belongs to someone it knows and whose face it is.

## Installation

### Requirements

- Python 3.9+
  - Versions below 3.9 have not been tested, and pickle module must be installed via pip.
- dilb

First, you need to install dilb. You can install it by following the instructions on the [here](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).

Then, you can install VisageSnap by using pip:

```bash
pip install visagesnap
```

## Usage

**Assign a label to the face**

You should assign a NameLabel to the face you want to classify first.

```python
vs = VisageSnap.Core()
people = ['Tom', 'Jerry']
# ['NameLabel1', 'NameLabel2', 'NameLabel3'...]

vs.set_label(people)
```

You can also do it like this so that assign a NameLabel and NumberLabel: _(NumberLabel MUST NOT BE -1)_

```python
people = {
    # 'NameLabel': NumberLabel
    'Tom': 0,
    'Jerry': 1
}
```

Put the picture files to be used during training in the directory. In this case, the file name follows the following rules:

`(NameLabel)-(Any character).extension`

> Tom-123.png<br>
> Tom-124.jpg<br>
> Tom-126.jpeg<br>
> Jerry-2.png<br>
> Jerry-3.png<br>
> Jerry-4.png<br>

**Recognize faces and train the model**

Before training, you need to load model.
You don't have a model? It's okay. It'll be created automatically if it doesn't exist.

```python
vs.load_model()
```

Train with the picture files in the directory.

```python
vs.train_labeled_data()
```

If you want to train with unlabeled data, you can also try to like this:

```python
vs.train_unlabeled_data()
```

**Identification predictions**

Put the picture files you want to predict into the directory.

```python
result = vs.predict_all()
print(result)
```

```python
{
   "target1.png": "Tom",
   "target2.jpeg": "Jerry",
   "target3.jpeg": ["Tom", "Jerry"], # multiple faces in one picture
   "target4.jpeg": None # If the face is unknown
}
```

**Full example**

You can see the full code [here](tests/test.py).

---

**To change the directory you work with**

You should put the picture files into configured directory, and also model file is stored in model directory.

```python
from VisageSnap import Directory

my_dir = Directory( "labeled_pic",
                    "unlabeled_pic",
                    "my_model.d",
                    "predict_dir")
vs.set_directory(my_dir)
```

_Default Directory:_

```python
{
    "labeled": "labeled",
    "unlabeled": "unlabeled",
    "model": "model"
}
```

### Others

**To get the face information**

You can get the faceObject by using `get_faceObject` method. The faceObject is a dataclass that contains the face information.

```python
from visagesnap import From
# From.LABEL, From.FILENAME
```

```python
face_tom = vs.get_faceObject(From.LABEL, "Tom")
face_tom = vs.get_faceObject(From.FILENAME, "Tom-123.png")
# face_tom = Face(label, encodings, filenames)

name: str = face_tom.label
encodings: NDArray = face_tom.encodings
filenames: list = face_tom.filenames
```

**To change prediction threshold**

```python
vs.threshold = 0.5  # Default: 0.48
```

## Acknowledgement

- [scikit-learn](https://scikit-learn.org/stable/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
