import VisageSnap

vs = VisageSnap.Core()
vs.set_label(["NY", "JY"])
result = vs.predict_all()
print(result)