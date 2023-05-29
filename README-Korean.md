# VisageSnap

![Release](https://shields.io/github/v/release/asheswook/VisageSnap?display_name=tag&sort=semver) ![build](https://img.shields.io/github/actions/workflow/status/asheswook/VisageSnap/docker-workflow.yml?branch=main)

[English](README.md) | **한국어**

사진에서 얼굴을 감지하고 준지도 학습 (반지도 학습)을 통해서 모델을 학습시키고, 얼굴을 분류합니다.

## Feature

- 사진에서 얼굴을 인식합니다.
- 준지도 학습을 통해 라벨링된 사진과 라벨링되지 않은 사진으로 모델을 학습시킬 수 있습니다.
- 해당하는 얼굴이 모르는 사람의 얼굴인지, 아는 사람이라면 누구인지를 예측합니다.

## Installation

### Requirements

- Python 3.9+
  - 3.9 이하 버전은 테스트되지 않았으며, pip를 통해 pickle 모듈을 설치해야 할 수 있습니다.
- dilb

먼저 dilb를 설치해야 합니다. [이 곳](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)의 지시에 따라 설치할 수 있습니다.

그 다음 pip를 이용해 VisageSnap를 설치합니다.

```bash
pip install visagesnap
```

## Usage

**얼굴에 라벨을 할당하기**

분류하기를 원하는 얼굴에 NameLabel을 할당해야 합니다.

```python
vs = VisageSnap.Core()
people = ['Tom', 'Jerry']
# ['NameLabel1', 'NameLabel2', 'NameLabel3'...]

vs.set_label(people)
```

또는 아래와 같이 NameLabel과 NumberLabel을 할당할 수도 있습니다: _(NumberLabel에는 절대 -1을 할당할 수 없습니다)_

```python
people = {
    # 'NameLabel': NumberLabel
    'Tom': 0,
    'Jerry': 1
}
```

학습에 사용할 사진 파일을 디렉토리에 넣어야 합니다. 이 때 파일 이름은 다음과 같은 규칙을 따릅니다:

`(NameLabel)-(아무 글자).확장자`

> Tom-123.png<br>
> Tom-124.jpg<br>
> Tom-126.jpeg<br>
> Jerry-2.png<br>
> Jerry-3.png<br>
> Jerry-4.png<br>

**얼굴을 인식하고 모델을 학습시키기**

학습하기 전에, 먼저 모델을 불러와야합니다. 모델이 없다구요? 상관 없습니다. 모델이 모델 디렉토리에 존재하지 않으면 자동으로 새로운 모델을 생성합니다.

```python
vs.load_model()
```

이제 디렉토리에 있는 사진 파일을 통해 모델을 학습시킵니다.

```python
vs.train_labeled_data()
```

만약 라벨링되지 않은 사진 파일로 모델을 학습시키고 싶다면 다음과 같이 할 수 있습니다.

```python
vs.train_unlabeled_data()
```

**얼굴 예측 (분류)**

예측하기를 원하는 사진 파일을 디렉토리에 넣고 다음과 같이 실행합니다.

```python
result = vs.predict_all()
print(result)
```

결과 값은 다음과 같습니다.

```python
{
   "target1.png": "Tom",
   "target2.jpeg": "Jerry",
   "target3.jpeg": ["Tom", "Jerry"], # 한 사진 안에 여러 사람이 있을 때
   "target4.jpeg": None # 모르는 사람일 떄
}
```

**전체 코드**

[예제 전체 코드](tests/test.py)

---

**디렉토리 설정**

사진 파일들은 설정된 디렉토리에 있어야 합니다. 또한 모델 파일도 설정된 모델 디렉토리에 저장되고 로드됩니다. 다음과 같이 디렉토리를 설정할 수 있습니다.

```python
from VisageSnap import Directory

my_dir = Directory( "labeled_pic",
                    "unlabeled_pic",
                    "my_model.d",
                    "predict_dir")
vs.set_directory(my_dir)
```

_기본 디렉토리:_

```python
{
    "labeled": "labeled",
    "unlabeled": "unlabeled",
    "model": "model"
}
```

### 기타 메소드

**얼굴의 정보 얻기**

`get_faceObject` 메소드를 이용하여 faceObject를 얻을 수 있습니다. faceObject는 얼굴 정보를 포함하고 있는 dataclass입니다.

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

**결정 임계값 (threshold) 변경하기**

```python
vs.threshold = 0.5  # Default: 0.48
```

## Acknowledgement

- [scikit-learn](https://scikit-learn.org/stable/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
