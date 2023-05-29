import VisageSnap

vs = VisageSnap.Core()
vs.load_model()
vs.set_label(["NY", "JY"])
vs.train_labeled_data()
result = vs.predict_all()
print(result)