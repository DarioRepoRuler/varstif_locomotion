from PIL import Image
import numpy as np
import trimesh

# Load the image
img = Image.open('/home/dspoljaric/Documents/TALocoMotion/envs/resources/unitree_go2/assets/height_map.png').convert('L')

# Downsample the image
img = img.resize((img.size[0] // 2, img.size[1] // 2))

# Convert the image data to a numpy array
data = np.asarray(img)

# Normalize the data
data = data / np.average(data)

# Create a grid of the same size as the image
x = np.arange(0, data.shape[0])
y = np.arange(0, data.shape[1])
x, y = np.meshgrid(x, y)

# Create vertices
vertices = np.column_stack([x.flatten(), y.flatten(), data.flatten()])

# Create faces
faces = []
for j in range(data.shape[1]-1):
    for i in range(data.shape[0]-1):
        faces.append([i+j*data.shape[0], i+1+j*data.shape[0], i+(j+1)*data.shape[0]])
        faces.append([i+1+j*data.shape[0], i+1+(j+1)*data.shape[0], i+(j+1)*data.shape[0]])

# Create the mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
print(f"Number of vertives: {len(vertices)}")
# Save the mesh to a file
mesh.export('/home/dspoljaric/Documents/TALocoMotion/envs/resources/unitree_go2/assets/terrain.stl')

# Load the mesh from the file
loaded_mesh = trimesh.load_mesh('/home/dspoljaric/Documents/TALocoMotion/envs/resources/unitree_go2/assets/terrain.stl')

# Visualize the mesh
loaded_mesh.show()