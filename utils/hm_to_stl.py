from PIL import Image
import numpy as np
import trimesh

# Load the image
img = Image.open('/home/dspoljaric/Documents/TALocoMotion/perlin_noise_with_steps.png').convert('L')

# Optionally downsample for a lower-resolution mesh (uncomment if needed)
img = img.resize((img.size[0] // 2, img.size[1] // 2))

# Get image dimensions 
width, height = img.size  
# Set pixel size and scaling factors (in meters per pixel)
pixel_size = 0.01
x_scale_factor = 1.0  # Adjust these factors as needed
y_scale_factor = 1.0

# Convert the image data to a numpy array
data = np.asarray(img)

# Normalize the data to the range [0, 1]
data = data / 255.0  # Assuming 8-bit grayscale image (0-255)

# Optionally set a maximum height in meters
max_height = 1  # Adjust this based on your desired terrain height
data *= max_height 

# Create a grid of the same size as the image, centered around (0, 0)
x = np.linspace(-width / 2 * pixel_size * x_scale_factor, (width / 2 - 1) * pixel_size * x_scale_factor, width) 
y = np.linspace(-height / 2 * pixel_size * y_scale_factor, (height / 2 - 1) * pixel_size * y_scale_factor, height)
x, y = np.meshgrid(x, y)

# Calculate the average Z height
average_z = np.mean(data)

# Create vertices with the Z coordinate offset by the average height
vertices = np.column_stack([x.flatten(), y.flatten(), (data - average_z).flatten()])

# Find the minimum z value of the top vertices
min_z = np.min(vertices[:, 2])

# Create a duplicate set of vertices at min_z
vertices_bottom = np.column_stack([x.flatten(), y.flatten(), np.full_like(data.flatten(), min_z)])

# Combine the top and bottom vertices
vertices = np.vstack([vertices, vertices_bottom])

# Create faces
faces = []
for j in range(data.shape[1] - 1):
    for i in range(data.shape[0] - 1):
        # Top faces
        faces.append([i + j * data.shape[0], i + 1 + j * data.shape[0], i + (j + 1) * data.shape[0]])
        faces.append([i + 1 + j * data.shape[0], i + 1 + (j + 1) * data.shape[0], i + (j + 1) * data.shape[0]])

        # Bottom faces
        offset = data.shape[0] * data.shape[1]  # Offset to index into the bottom vertices
        faces.append([i + j * data.shape[0] + offset, i + (j + 1) * data.shape[0] + offset, i + 1 + j * data.shape[0] + offset])
        faces.append([i + 1 + j * data.shape[0] + offset, i + (j + 1) * data.shape[0] + offset, i + 1 + (j + 1) * data.shape[0] + offset])

# Add vertical faces around the perimeter
for i in range(data.shape[0] - 1):
    # Front face
    faces.append([i, i + 1, i + offset])
    faces.append([i + 1, i + 1 + offset, i + offset])

    # Back face
    faces.append([i + (data.shape[1] - 1) * data.shape[0], i + 1 + (data.shape[1] - 1) * data.shape[0], i + (data.shape[1] - 1) * data.shape[0] + offset])
    faces.append([i + 1 + (data.shape[1] - 1) * data.shape[0], i + 1 + (data.shape[1] - 1) * data.shape[0] + offset, i + (data.shape[1] - 1) * data.shape[0] + offset])

for j in range(data.shape[1] - 1):
    # Left face
    faces.append([j * data.shape[0], (j + 1) * data.shape[0], j * data.shape[0] + offset])
    faces.append([(j + 1) * data.shape[0], (j + 1) * data.shape[0] + offset, j * data.shape[0] + offset])

    # Right face
    faces.append([(j + 1) * data.shape[0] - 1, (j + 2) * data.shape[0] - 1, (j + 1) * data.shape[0] - 1 + offset])
    faces.append([(j + 2) * data.shape[0] - 1, (j + 2) * data.shape[0] - 1 + offset, (j + 1) * data.shape[0] - 1 + offset])

# Create the mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Simplify the mesh
target_face_count = 200000
mesh = mesh.simplify_quadratic_decimation(target_face_count)

# Save the mesh to a file
mesh.export('/home/dspoljaric/Documents/TALocoMotion/envs/resources/unitree_go2/assets/stairs.stl')



mesh.show()
