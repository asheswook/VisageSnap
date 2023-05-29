import sys
import os
sys.path.append(sys.path[0].replace("tests", ""))
print(sys.path)
import VisageSnap

### ------------------------------------
### DO NOT USE THE CODE ABOVE THIS LINE!
### ------------------------------------

vs = VisageSnap.Core()
vs.load_model()
vs.set_label(["NY", "JY"])
vs.train_labeled_data()
result = vs.predict_all()
print(result)