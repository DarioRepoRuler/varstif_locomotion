import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.graphs_gen import load_tensor_from_csv


class VideoPlotCombiner:
    def __init__(self, video_paths, data_files, output_video_path, window_size=50, fps=25, time_scale=0.02):
        self.video_path = video_paths
        self.tensor_file = data_files
        self.output_video_path = output_video_path
        self.window_size = window_size
        self.fps = fps
        self.time_scale = time_scale
        
        # Load tensors
        self.Pgains = torch.stack([load_tensor_from_csv('p_gains_values', filename=file) for file in data_files])
        self.Position_errors = torch.stack([load_tensor_from_csv('position_errors', filename=file) for file in data_files])
        self.Power = torch.stack([load_tensor_from_csv('power', filename=file) for file in data_files])
        
        # Open the video files
        self.caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
        for cap in self.caps:
            if not cap.isOpened():
                print(f"Error: Could not open video.")
                exit()

        # Video dimensions (assuming all videos have the same dimensions)
        self.frame_width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set up the output video (Fix width and height calculation)
        self.plot_width = 700
        self.video_crop_size = 250
        combined_width = self.video_crop_size + self.plot_width  # Two videos side-by-side + space for plot
        combined_height = self.video_crop_size * len(video_paths)  # Stack videos vertically
        #print(f"Combined width: {combined_width}, Combined height: {combined_height}")

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # For MP4 output
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (combined_width, combined_height))

        # Buffers for sliding window
        self.pgains_window = []
        self.position_errors_window = []
        self.power_window = []
        self.time_window = []

        # Set up plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()  # Enable interactive plotting

    def update_buffers(self, t):
        current_pgains = self.Pgains[:,t, :].cpu().numpy()
        current_position_errors = self.Position_errors[:,t, :].cpu().numpy()
        current_power = self.Power[:,t, :].cpu().numpy()
        self.pgains_window.append(current_pgains)
        self.position_errors_window.append(current_position_errors)
        self.power_window.append(current_power)
        self.time_window.append(t * self.time_scale)  # Scale the time by 0.02

        if len(self.pgains_window) > self.window_size:
            self.pgains_window.pop(0)
            self.position_errors_window.pop(0)
            self.power_window.pop(0)
            self.time_window.pop(0)

    def update_plots(self):
        pgains_array = np.array(self.pgains_window)
        power_array = np.array(self.power_window)

        self.ax1.clear()
        self.ax2.clear()
        # Generate distinct base colors for robots
        num_robots = len(self.tensor_file)
        base_colors = plt.cm.tab10(np.linspace(0, 1, num_robots))  # Use tab10 colormap for distinct colors

        for robot in range(len(self.tensor_file)):
            position_based= (pgains_array[0, robot, 0] == pgains_array[0, robot, 3] and pgains_array[0, robot, 0] == pgains_array[0, robot, 6] and pgains_array[0, robot, 0] == pgains_array[0, robot, 11])
            if position_based:
                    self.ax1.plot( self.time_window, pgains_array[:, robot, 0], label=f"Robot {robot + 1}", color=base_colors[robot], linewidth=2.5)
            else: 
                for leg in range(4):
                    leg_color = base_colors[robot] * (1 - 0.1 * leg)
                    self.ax1.plot(self.time_window, pgains_array[:, robot, leg * 3], label=f"Robot {robot + 1}, Leg {leg + 1}", color=leg_color)
            self.ax1.legend(loc="upper right", ncol=2, fontsize=8)
            self.ax1.set_ylabel("P Gains")

            self.ax2.plot(self.time_window, power_array[:, robot, 0], label=f"Power [W] Robot {robot + 1}", color=base_colors[robot], linewidth=2.5)
            self.ax2.legend(loc="upper right", ncol=2, fontsize=8)

            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Power [W]")

            self.fig.canvas.draw()

    def capture_and_process_frame(self, t):
        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video.")
                return False
            # Resize each frame to 250x250
            frame_resized = cv2.resize(frame, (self.video_crop_size, self.video_crop_size))
            frames.append(frame_resized)

        combined_video_frame = np.vstack(frames)

        # Convert plot to numpy array
        plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plot_img_resized = cv2.resize(plot_img, (self.plot_width, self.video_crop_size*len(self.video_path)))  # Resize plot for stacked videos
        final_frame = np.hstack([combined_video_frame, plot_img_resized])
        #print(f"PLot shape: {plot_img_resized.shape}")
        #print(f"Combined frame shape: {combined_video_frame.shape}")
        #print(f"Final frame shape: {final_frame.shape}")
        self.out.write(final_frame)
        return True

    def process_videos_and_plots(self):
        num_frames = self.Pgains.shape[1]

        for t in range(num_frames):
            self.update_buffers(t)
            self.update_plots()
            if not self.capture_and_process_frame(t):
                break
            plt.pause(1 / self.fps)

    def save_and_cleanup(self):
        for cap in self.caps:
            cap.release()
        self.out.release()
        plt.close()
        print(f"Video saved at: {self.output_video_path}")


if __name__ == "__main__":
    video_paths = ['/home/dario/Documents/TALocoMotion/outputs/videos/stiffness_test_direction_2024-11-25_10-20-58.mp4', '/home/dario/Documents/TALocoMotion/outputs/videos/stiffness_test_direction_2024-11-24_09-58-49.mp4','/home/dario/Documents/TALocoMotion/outputs/videos/stiffness_test_direction_2024-11-25_14-03-45.mp4']
    tensor_files = ['stiffness_test_direction_2024-11-25_10-20-58.csv','stiffness_test_direction_2024-11-24_09-58-49.csv','stiffness_test_direction_2024-11-25_14-03-45.csv' ]
    output_video_path = '/home/dario/Videos/combined_video.mp4'

    combiner = VideoPlotCombiner(video_paths, tensor_files, output_video_path)
    combiner.process_videos_and_plots()
    combiner.save_and_cleanup()
