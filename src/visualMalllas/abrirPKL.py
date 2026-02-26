import pickle
import trimesh
import numpy as np

# Ruta a tu archivo (ajustada a tu sistema)
path = './mallas/mallas_pkl/female.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Extraemos vértices y caras
vertices = data['v_template']
faces = np.array(data['f'], dtype=np.int32)

# --- 2. Crear los objetos de visualización ---

# Primero, creamos la malla base en memoria.
# No la vamos a mostrar, pero la necesitamos para que trimesh calcule
# automáticamente dónde están las "uniones" (edges) únicas.
base_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# A) Crear la visualización de los PUNTOS (Vértices)
# Los haremos de color ROJO vivo para que destaquen.
# El formato de color es [R, G, B, Alpha(transparencia)] de 0 a 255.
red_color = [255, 0, 0, 255] 
points_viz = trimesh.points.PointCloud(vertices, colors=[red_color for _ in vertices])

# B) Crear la visualización de las UNIONES (Aristas/Wireframe)
# 'base_mesh.edges_unique' nos da las parejas de índices que forman una línea.
# Usamos trimesh.load_path para convertir esas parejas en líneas 3D visibles.
# Las haremos de color NEGRO o gris oscuro.
lines_viz = trimesh.load_path(base_mesh.vertices[base_mesh.edges_unique])
for entity in lines_viz.entities:
    entity.color = [50, 50, 50, 255] # Gris oscuro

# --- 3. Montar la escena y mostrarla ---

# Creamos una escena combinando los puntos y las líneas
scene = trimesh.Scene([points_viz, lines_viz])

# Opcional: Desactivar los ejes si molestan
# scene.show(flags={'axis': False}) 

print("Abriendo visualización de Puntos + Wireframe...")
print("Usa el ratón para rotar y hacer zoom.")
scene.show()