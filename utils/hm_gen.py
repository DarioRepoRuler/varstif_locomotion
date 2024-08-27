import numpy as np
from PIL import Image
from noise import pnoise2
import os
from scipy.ndimage import gaussian_filter


class HeightMapGenerator:
    def __init__(self, height=520, width=520):
        self.height = height
        self.width = width
        self.height_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.height_map[:] = 255//2
        
    def generate_perlin_noise(self, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, start_x=0, start_y=0, end_x=100, end_y=100 ,maximum = 255):
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
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min()) * maximum
        height_map = height_map.astype(np.uint8)
        self.height_map[start_x:end_x, start_y:end_y] += height_map[start_x:end_x, start_y:end_y]
        return height_map
    
    def generate_gaussian_hills(self, num_hills, hill_height=255, hill_radius=50, start_x=0, start_y=0, end_x=150, end_y=150):
        """
        Generate Gaussian hills on the height map.

        :param num_hills: Number of hills to generate
        :param hill_height: Height of each hill
        :param hill_radius: Radius of each hill
        :return: numpy array representing the height map with Gaussian hills
        """
        width = end_y-start_y
        height = end_x-start_x
        height_map_with_hills = np.zeros((width, height ), dtype=np.uint8)  # Create a copy of the original height map

        # center_x = np.random.randint(0, width-hill_radius, size = num_hills)
        # center_y = np.random.randint(0, height-hill_radius, size=num_hills)
        center_x = np.random.choice(np.arange(0, width-hill_radius), size=num_hills)
        center_y = np.random.choice(np.arange(0, height-hill_radius), size=num_hills)

        for i in range(num_hills):
            # Randomly select the center of the hill
            

            # Generate the hill using Gaussian function
            for y in range(0, width-1):
                for x in range(0, height-1):
                    distance = np.sqrt((x - center_x[i]) ** 2 + (y - center_y[i]) ** 2)
                    if distance < hill_radius:
                        height_map_with_hills[y][x] += hill_height * np.exp(-((distance ** 2) / ((hill_radius**2 /4))))

        # Clip values to ensure they are within valid range
        height_map_with_hills = np.clip(height_map_with_hills, 0, 255)
        self.height_map[ start_y:end_y, start_x:end_x] += height_map_with_hills
        return height_map_with_hills
    
    def generate_linear_steps(self, num_steps=10, corner_x=0, corner_y=0, staircase_width=10, staircase_height=10, direction='horizontal_right'):
        """
        Generate a height map with linear steps from left to right or right to left.

        Parameters:
        - num_steps (int): The number of steps in the staircase. Default is 10.
        - corner_x (int): The x-coordinate of the top-left corner of the staircase. Default is 0.
        - corner_y (int): The y-coordinate of the top-left corner of the staircase. Default is 0.
        - staircase_width (int): The width of each step in the staircase. Default is 10.
        - staircase_height (int): The height of the staircase. Default is 10.
        - direction (str): The direction of the staircase. Can be 'horizontal_right' or 'horizontal_left'. Default is 'horizontal_right'.

        Returns:
        - height_map (numpy.ndarray): The generated height map with linear steps.

        Raises:
        - ValueError: If the number of steps is less than 1.
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        
        height_map = np.zeros((self.height, self.width), dtype=np.uint8)
        step_width = staircase_width // num_steps

        if direction == 'horizontal_right':
            start_y = corner_y
            end_y = start_y + staircase_height
            
            for step in range(num_steps):
                grayscale_value = int((step / (num_steps - 1)) * 255)
                
                start_x = corner_x + step * step_width
                end_x = start_x + step_width 
                
                height_map[start_y:end_y, start_x:end_x] = grayscale_value

        elif direction == 'horizontal_left':
            start_y = corner_y
            end_y = start_y + staircase_height
            
            for step in range(num_steps):
                grayscale_value = int((step / (num_steps - 1)) * 255)
                
                start_x = corner_x + (num_steps-step) * step_width
                end_x = start_x + step_width 
                
                height_map[start_y:end_y, start_x:end_x] = grayscale_value
        
        self.height_map += height_map
        return height_map
    
    def pyramid(self, num_steps=5, corner_x=0, corner_y=0, staircase_width=10, maximum=255):
        """
        Generate a pyramid on the height map.

        Args:
            num_steps (int): The number of steps in the pyramid. Default is 5.
            corner_x (int): The x-coordinate of the top-left corner of the pyramid. Default is 0.
            corner_y (int): The y-coordinate of the top-left corner of the pyramid. Default is 0.
            staircase_width (int): The width of each step in the pyramid. Default is 10.
            maximum (int): The maximum grayscale value of the pyramid. Default is 255.

        Returns:
            numpy.ndarray: The generated height map with the pyramid.
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        
        height_map = np.zeros((self.height, self.width), dtype=np.uint8)
        step_width = staircase_width // (2 * num_steps)

        for step in range(num_steps):
            grayscale_value = int((step / (num_steps - 1)) * maximum)

            start_x = corner_x + step * step_width
            end_x = corner_x + staircase_width - step * step_width
            start_y = corner_y + step * step_width
            end_y = corner_y + staircase_width - step * step_width

            height_map[start_y:end_y, start_x:end_x] = grayscale_value

        self.height_map -= height_map
        return height_map
    
    def gen_multiple_pyramids(self, num_pyramids=5, num_steps=5, staircase_width=10):
        """
        Generate multiple pyramids on the height map.

        Args:
            num_pyramids (int): The number of pyramids to generate. Default is 5.
            num_steps (int): The number of steps in each pyramid. Default is 5.
            staircase_width (int): The width of each step in each pyramid. Default is 10.
        """
        corner_x = np.random.randint(0, self.width-1, size=num_pyramids)
        corner_y = np.random.randint(0, self.height-1, size=num_pyramids)
        for i in range(num_pyramids):
            corner_x = np.random.randint(0, self.width-1)
            corner_y = np.random.randint(0, self.height-1)
            self.pyramid(num_steps, corner_x, corner_y, staircase_width)

    def stripes(self, num_stripes=5, stripe_width=10, height=200, direction='horizontal', win_start_x=0, win_start_y=0, win_end_x=100, win_end_y=100):
        """
        Generate a height map with stripes.

        Args:
            num_stripes (int): The number of stripes to generate. Default is 5.
            stripe_width (int): The width of each stripe. Default is 10.
            height (int): The grayscale value of the stripes. Default is 200.
            direction (str): The direction of the stripes. Can be 'horizontal' or 'vertical'. Default is 'horizontal'.
            win_start_x (int): The starting x-coordinate of the window to apply the stripes. Default is 0.
            win_start_y (int): The starting y-coordinate of the window to apply the stripes. Default is 0.
            win_end_x (int): The ending x-coordinate of the window to apply the stripes. Default is 100.
            win_end_y (int): The ending y-coordinate of the window to apply the stripes. Default is 100.

        Returns:
            numpy.ndarray: The generated height map with stripes.
        """
        height_map = np.zeros((self.height, self.width), dtype=np.uint8)

        if direction == 'horizontal':
            for i in range(num_stripes):
                if i % 2 == 0:
                    grayscale_value = height
                else:
                    grayscale_value = 0
                start_x = i * stripe_width
                end_x = start_x + stripe_width
                height_map[:, start_x:end_x] = grayscale_value
        elif direction == 'vertical':
            for i in range(num_stripes):
                if i % 2 == 0:
                    grayscale_value = height
                else:
                    grayscale_value = 0
                start_y = i * stripe_width
                end_y = start_y + stripe_width
                height_map[start_y:end_y, :] = grayscale_value

        self.height_map[win_start_x:win_end_x, win_start_y:win_end_y] += height_map[win_start_x:win_end_x, win_start_y:win_end_y]
        return height_map


    def blur_height_map(self, sigma=1, start_x=0, start_y=0, end_x=100, end_y=100):
        """
        Perform a Gaussian blur on the height map.

        :param sigma: Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
        """
        self.height_map[start_x:end_x, start_y:end_y] = gaussian_filter(self.height_map[start_x:end_x, start_y:end_y], sigma=sigma)
        #print(gaussian_filter(self.height_map[start_x:end_x, start_y:end_y], sigma=sigma).dtype)



    def save_height_map_as_image(self, filename):
        """
        Save the height map as a greyscale image.
        
        :param height_map: numpy array representing the height map
        :param filename: Filename to save the image(it will be stored in the current working directory)
        """
        image = Image.fromarray(self.height_map, mode='L')
        print(os.getcwd())
        file_path= os.path.join(os.getcwd(),'envs', 'resources','unitree_go2', 'assets', filename) 
        image.save(file_path)

def main():
    # Define the size of the height map
    width = 250
    height = 250
    hm_gen = HeightMapGenerator(height, width)
    
    # Generate the Perlin noise height map
    hm_gen.generate_perlin_noise(start_x=130, start_y=130, end_x=250, end_y=250, maximum=100) 
    
    # Generate Pyramids
    hm_gen.pyramid(5, 20, 80, 50, maximum=30)
    hm_gen.pyramid(5, 10, 10, 50,maximum=30)
    hm_gen.pyramid(5, 60, 50, 50, maximum=30)
    
    # Generate checkerboard pattern
    hm_gen.stripes(25, 10, height=5, direction='vertical', win_start_x=0, win_start_y=130, win_end_x=130, win_end_y=250)
    hm_gen.stripes(25, 10, height=5, direction='horizontal', win_start_x=0, win_start_y=130, win_end_x=130, win_end_y=250)

    # Generate gaussian hills
    hm_gen.generate_gaussian_hills(8, 40, 30, start_x=0, start_y=130, end_x=130, end_y=250)

    #
    hm_gen.blur_height_map(sigma=3, start_x=120, start_y=120, end_x=250, end_y=140)
    hm_gen.blur_height_map(sigma=3, start_x=120, start_y=120, end_x=140, end_y=250)
    hm_gen.blur_height_map(sigma=3, start_x=120, start_y=0, end_x=140, end_y=250)

    # Define the filename
    filename = 'new_terrain.png'
    
    # Save the height map with steps as an image
    hm_gen.save_height_map_as_image(filename)
    print(f'Height map saved as {filename}')

if __name__ == "__main__":
    main()