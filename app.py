import streamlit as sts
import numpy as np
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from sklearn.metrics import roc_auc_score

VOCAB_LIST = ['healthy', 'multiple_diseases', 'rust', 'scab']

sts.title("Apple Leaf Disease Detector")

sts.text("Submit a apple leaf image and the app will detect its disease(if any)")


file = sts.file_uploader("Pick a file")


model_path = Path("model")

def get_x(f):
    return f"{path}/{f['image_id']}.jpg"

def get_y(f):
    return f['label']

def total_error_metric(preds, targets, labels = range(len(VOCAB_LIST))):
    #one-hot encoded targets
    targets = np.eye(4)[targets]
    return np.mean([roc_auc_score(targets[:,i],preds[:,i]) for i in labels])

def healthy_roc_Auc(*args):
    return total_error_metric(*args, labels = [0])

def multiple_diseases_roc_Auc(*args):
    return total_error_metric(*args, labels = [1])

def rust_roc_Auc(*args):
    return total_error_metric(*args, labels = [2])

def scab_roc_Auc(*args):
    return total_error_metric(*args, labels = [3])

if file is not None:
    if sts.button("Detect"):

        learn_inf = load_learner(model_path/"base_model_export.pkl", cpu=True)

        output = learn_inf.predict(file.read())[0]

        if str(output) == "healthy":
            sts.header(f"The Apple Leaf is : {str(output).capitalize()}")
        else:
            sts.header(f"The Apple Leaf has : {str(output).capitalize()} disease")