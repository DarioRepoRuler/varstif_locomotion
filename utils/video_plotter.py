import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.graphs_gen import load_tensor_from_csv


class VideoPlotCombiner:
    def __init__(self, video_path, tensor_file, output_video_path, window_size=50, fps=25, time_scale=0.02):
        """
        Initializes the class with necessary parameters.
        
        :param video_path: Path to the video file to read
        :param tensor_file: Path to the CSV file containing tensors
        :param output_video_path: Path to the output video file to save
        :param window_size: Number of time steps to show in the sliding window
        :param fps: Frames per second for video and plot update
        :param time_scale: Time scale for the x-axis in the plots (e.g., 0.02 seconds per step)
        """
        self.video_path = video_path
        self.tensor_file = tensor_file
        self.output_video_path = output_video_path
        self.window_size = window_size
        self.fps = fps
        self.time_scale = time_scale
        
        # Load tensors
        self.Pgains = load_tensor_from_csv('p_gains_values', filename=tensor_file)
        self.Position_errors = load_tensor_from_csv('position_errors', filename=tensor_file)
        self.Power = load_tensor_from_csv('power', filename=tensor_file)
        
        # Open the video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        # Set up the output video
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width * 2, self.frame_height))
        
        # Buffers to hold the sliding window of data
        self.pgains_window = []
        self.position_errors_window = []
        self.power_window = []
        self.time_window = []
        
        # Set up the plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()  # Enable interactive mode for live plotting

    def update_buffers(self, t):
        """
        Updates the sliding window buffers with the new data for the current timestep.
        """
        current_pgains = self.Pgains[t, :].cpu().numpy()
        current_position_errors = self.Position_errors[t, :].cpu().numpy()
        current_power = self.Power[t, :].cpu().numpy()
        
        # Update the buffers with new data
        self.pgains_window.append(current_pgains)
        self.position_errors_window.append(current_position_errors)
        self.power_window.append(current_power)
        self.time_window.append(t * self.time_scale)  # Scale the time by 0.02

        # Keep only the last `window_size` elements
        if len(self.pgains_window) > self.window_size:
            self.pgains_window.pop(0)
            self.position_errors_window.pop(0)
            self.power_window.pop(0)
            self.time_window.pop(0)

    def update_plots(self):
        """
        Updates the plots based on the data in the sliding window buffers.
        """
        # Convert buffers to numpy arrays for plotting
        pgains_array = np.array(self.pgains_window)
        position_errors_array = np.array(self.position_errors_window)
        power_array = np.array(self.power_window)
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Update P Gains plot
        for leg in range(4):
            self.ax1.plot(self.time_window, pgains_array[:, leg * 3], label=f"Leg {leg}")
        self.ax1.legend(loc="upper right", ncol=2, fontsize=8)
        self.ax1.set_title("P Gains over Time")
        self.ax1.set_xlabel("Time (s) (Last second)")
        self.ax1.set_ylabel("P Gains")

        # Update Power plot
        self.ax2.plot(self.time_window, power_array[:, 0], label="Power [W]")
        self.ax2.legend(loc="upper right", ncol=2, fontsize=8)
        self.ax2.set_title("Power [W]")
        self.ax2.set_xlabel("Time (s) (Last second)")
        self.ax2.set_ylabel("Power [W]")

        # Draw the updated plots
        self.fig.canvas.draw()

    def capture_and_process_frame(self, t):
        """
        Captures a video frame, combines it with the plot, and writes the result to the output video.
        """
        # Capture the video frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            return False

        # Convert the frame from BGR (OpenCV) to RGB (matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the plot into a numpy array (in RGB format)
        plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # Resize plot image to match the video frame height
        plot_img_resized = cv2.resize(plot_img, (self.frame_width, self.frame_height))

        # Combine the video frame with the plot (side-by-side)
        combined_frame = np.hstack((frame, plot_img_resized))

        # Write the combined frame to the output video
        self.out.write(combined_frame)

        return True

    def process_video_and_plots(self):
        """
        Processes each frame of the video, updating the plot and writing the combined output.
        """
        num_frames = self.Pgains.shape[0]
        
        for t in range(num_frames):
            # Update the sliding window buffers with the new data
            self.update_buffers(t)

            # Update the plots with the current data
            self.update_plots()

            # Capture and combine the video frame with the plot, then write it to the output
            if not self.capture_and_process_frame(t):
                break

            # Add a small pause for visualization (if running interactively)
            plt.pause(1 / self.fps)

    def save_and_cleanup(self):
        """
        Finalizes the video saving and releases resources.
        """
        # Release resources
        self.cap.release()
        self.out.release()
        plt.close()

        print(f"Video saved at: {self.output_video_path}")


# Example usage
if __name__ == "__main__":
    video_path = '/home/dario/Videos/tracking_camera.mp4'
    tensor_file = 'stiffness_results_2024-11-25_10-20-58.csv'
    output_video_path = '/home/dario/Videos/combined_video.avi'

    combiner = VideoPlotCombiner(video_path, tensor_file, output_video_path)
    combiner.process_video_and_plots()
    combiner.save_and_cleanup()
