# VisageSnap

Recognizes faces and trains models, brings in the pictures and provides identification predictions and face classification. It also performs semi-supervised learning.

## Feature

-   Recognize faces.
-   Train the model through semi-supervised learning with labeled or unlabeled pictures.
-   It provides identification predictions and face classification for pictures.

## Usage

**Assign a label to the face**

You should assign a NameLabel to the face you want to classify first. _(NumberLabel MUST NOT BE -1)_

```python
vs = VisageSnap.Core()
people = ['Tom', 'Jerry']
# ['NameLabel1', 'NameLabel2', 'NameLabel3'...]

vs.set_label(people)
```

You can also do it like this so that assign a NameLabel and NumberLabel:

```python
vs = VisageSnap.Core()
people = {
    # 'NameLabel': NumberLabel
    'Tom': 0,
    'Jerry': 1
}

vs.set_label(people)
```

Put the picture files to be used during training in the directory. In this case, the file name follows the following rules:

`(NameLabel)-(Any character).extension`

> Tom-123.png<br>
> Tom-124.jpg<br>
> Tom-126.jpeg<br>
> Jerry-2.png<br>
> Jerry-3.png<br>
> Jerry-4.png<br>

---

**Recognize faces and train the model**

Train with the picture files in the directory.

```
vs.train_labeled_data()
```

If you want to train with unlabeled data, you can also try to like this:

```
vs.train_unlabeled_data()
```

---

**Identification predictions**

Put the picture files you want to predict into the directory.

```python
result = vs.predict_all()
print(result)

# Result:
#{
#   "target1.png": "Tom",
#   "target2.jpeg": "Jerry",
#   "target3.jpeg": "Jerry"
#}
```

**To change the directory you work with**

You should put the picture files into configured directory, and also model file is stored in model directory.

```python
vs.set_directory({
    "labeled": "labeled_pic",
    "unlabeled": "unlabeled_pic",
    "model": "my_model.d"
})
```

_Default Directory:_

```python
{
    "labeled": "labeled",
    "unlabeled": "unlabeled",
    "model": "model"
}
```
