import numpy as np
import trimesh

# 1. Cargar el NPZ
datos = np.load('./mallas/datos_npz/female.npz')

v_template = datos['v_template']
shapedirs = datos['shapedirs']
caras = datos['f']

# 2. VAMOS A CREAR UN CUERPO NUEVO
# Creamos 10 números aleatorios (betas) para cambiar la forma del cuerpo
# En tu proyecto, estos números vendrán de tu red neuronal o dataset
betas = np.zeros(10)
betas[0] = -2.0  # El primer parámetro suele controlar la altura/peso
betas[1] = 5.0 # El segundo suele controlar la musculatura/delgadez

# La fórmula mágica: Cuerpo = Base + (Direcciones * Parámetros)
# Esto aplica los cambios de forma a los puntos originales
puntos_modificados = v_template + np.dot(shapedirs[:, :, :10], betas)

# 3. Visualizar el resultado modificado
mesh = trimesh.Trimesh(vertices=puntos_modificados, faces=caras)

mesh.visual.face_colors = [200, 200, 200, 255] # Gris claro

mesh.show()