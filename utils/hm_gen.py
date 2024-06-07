import numpy as np
from PIL import Image
from noise import pnoise2
import os

class HeightMapGenerator:
    def __init__(self, height=520, width=520):
        self.height = height
        self.width = width
        self.height_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        
    def generate_perlin_noise(self, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        Generate a height map using Perlin noise.

        :param width: Width of the height map
        :param height: Height of the height map
        :param scale: Scale of the noise
        :param octaves: Number of octaves
        :param persistence: Persistence of the noise
        :param lacunarity: Lacunarity of the noise
        :return: numpy array representing the height map
        """
        height_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                height_map[y][x] = pnoise2(x / scale, 
                                        y / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=self.width, 
                                        repeaty=self.height, 
                                        base=42)
        
        # Normalize the result to be between 0 and 255
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min()) * 255
        height_map = height_map.astype(np.uint8)
        self.height_map = height_map
        return height_map
    
    def generate_gaussian_hills(self, num_hills, hill_height=255, hill_radius=50):
        """
        Generate Gaussian hills on the height map.

        :param num_hills: Number of hills to generate
        :param hill_height: Height of each hill
        :param hill_radius: Radius of each hill
        :return: numpy array representing the height map with Gaussian hills
        """
        height_map_with_hills = np.zeros((self.height, self.width), dtype=np.float32)  # Create a copy of the original height map

        for _ in range(num_hills):
            # Randomly select the center of the hill
            center_x = np.random.randint(0, self.width-1)
            center_y = np.random.randint(0, self.height-1)

            # Generate the hill using Gaussian function
            for y in range(0, self.width-1):
                for x in range(0, self.height-1):
                    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance < hill_radius:
                        height_map_with_hills[y][x] += hill_height * np.exp(-((distance ** 2) / ((hill_radius**2 /4))))

        # Clip values to ensure they are within valid range
        height_map_with_hills = np.clip(height_map_with_hills, 0, 255)
        self.height_map = height_map_with_hills
        return height_map_with_hills
    
    def generate_linear_steps(self, num_steps=10):
        """
        Generate a height map with linear steps from left to right.
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        
        height_map = np.zeros((self.height, self.width), dtype=np.uint8)
        step_width = self.width // num_steps
        
        for step in range(num_steps):
            grayscale_value = int((step / (num_steps - 1)) * 255)
            start_x = step * step_width
            end_x = start_x + step_width if step != num_steps - 1 else self.width
            height_map[:, start_x:end_x] = grayscale_value
        
        self.height_map = height_map
        return height_map


    def save_height_map_as_image(self, filename):
        """
        Save the height map as a greyscale image.
        
        :param height_map: numpy array representing the height map
        :param filename: Filename to save the image(it will be stored in the current working directory)
        """
        image = Image.fromarray(self.height_map, mode='L')
        file_path= os.path.join(os.getcwd(), filename) 
        image.save(file_path)

def main():
    # Define the size of the height map
    width = 520
    height = 520
    hm_gen = HeightMapGenerator(height, width)
    
    # Generate the Perlin noise height map
    #hm_gen.generate_perlin_noise()
    
    # Introduce steps to the height map
    hm_gen.generate_linear_steps(num_steps=5)
    
    # Define the filename
    filename = 'perlin_noise_with_steps.png'
    
    # Save the height map with steps as an image
    hm_gen.save_height_map_as_image(filename)
    print(f'Height map with steps saved as {filename}')

if __name__ == "__main__":
    main()