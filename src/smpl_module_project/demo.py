import os
import pickle
import trimesh

import torch
import numpy as np
import pandas as pd

from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from evaluate import evaluate_mae

from dotenv import load_dotenv
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

measurer = MeasureBody("smpl")
measurer2 = MeasureBody("smpl")

gender = "FEMALE"
betas = torch.tensor([-0.146978404,0.156539184,-1.948303312,0.003441101,-0.80102012,-0.068591649,0.113341932,-1.712147338,0.815856777,-0.254938078], dtype=torch.float32).unsqueeze(0)
measurer.from_body_model(gender=gender, shape=betas)
print(measurer.verts.shape) 

verts = measurer.verts  # (6890, 3)

# Faces de SMPL (si measurer las provee)
faces = measurer.faces if hasattr(measurer, "faces") else smpl_faces()  # (13776, 3)

# Crear malla
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# Mostrar
mesh.show()

measurement_names = measurer.all_possible_measurements # or chose subset of measurements 
measurer.measure(measurement_names) 
measurer.label_measurements(STANDARD_LABELS) 
print(measurer.measurements)
