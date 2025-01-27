from utils.graphs_gen import *
import os

import matplotlib.patches as mpatches

# File names and labels
filenames = [
    'stiffness_test_direction1_2024-11-24_09-58-49.csv',
    'stiffness_test_direction1_2024-11-25_14-03-45.csv',
    'stiffness_test_direction1_2024-11-25_10-20-58.csv'
]
labels = ['P50', 'P20', 'PLS']
legs = ['FR', 'FL', 'RR', 'RL']
# Initialize p_gains dictionary
p_gains = {label: [] for label in labels}

# Load data
for label, filename in zip(labels, filenames):
    p_gains[label] = load_tensor_from_csv('p_gains_values', filename)
print(p_gains['P50'].shape)
p_gains['P50'] = p_gains['P50'][:,0]
p_gains['P20'] = p_gains['P20'][:,0]
p_gains['PLS'] = p_gains['PLS'][:,::3]
# Create the time axis
timesteps = 0.02  # Time step in seconds
time = [i * timesteps for i in range(len(p_gains['P50']))]  # Assuming all files have the same length

# Plot the p-gains
plt.figure(figsize=(10, 6))
for label in labels:
    if label == 'PLS':
        for i in range(p_gains[label].shape[1]):
            plt.plot(time, p_gains[label][:,i], label=f'PLS: leg {legs[i]}')
    else:
        plt.plot(time, p_gains[label], label=label)

# Add plot details
plt.xlabel('Time (seconds)', fontsize=16)
plt.ylabel('P-Gains', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()

# Save or display the plot
plt.savefig('p_gains_plot.png')  # Save the plot as a file
plt.show()  # Display the plot



# video_filenames = [f'/home/dario/Desktop/Force Push evaluation/Recording/Push rear/stiffness_test_2024-11-24_09-58-49.mp4', f'/home/dario/Desktop/Force Push evaluation/Recording/Push front/stiffness_test_2024-11-25_14-03-45.mp4', f'/home/dario/Desktop/Force Push evaluation/Recording/Push left side/stiffness_test_2024-11-25_10-20-58.mp4']


# import cv2
# import os

# # Video filenames
# video_filenames = [
#     '/home/dario/Desktop/Force Push evaluation/Recording/Push rear/stiffness_test_2024-11-24_09-58-49.mp4',
#     '/home/dario/Desktop/Force Push evaluation/Recording/Push rear/stiffness_test_2024-11-25_14-03-45.mp4',
#     '/home/dario/Desktop/Force Push evaluation/Recording/Push rear/stiffness_test_2024-11-25_10-20-58.mp4'
# ]
# # video_filenames = [
# #     '/home/dario/Desktop/Force Push evaluation/Recording/Push front/stiffness_test_direction1_2024-11-24_09-58-49.mp4',
# #     '/home/dario/Desktop/Force Push evaluation/Recording/Push front/stiffness_test_direction1_2024-11-25_14-03-45.mp4',
# #     '/home/dario/Desktop/Force Push evaluation/Recording/Push front/stiffness_test_direction1_2024-11-25_10-20-58.mp4'   
# # ]

# # Timesteps in seconds for frame extraction
# timesteps = [9.1, 9.2, 9.5, 9.8]

# # Output folder for extracted frames
# output_folder = "/home/dario/Desktop/ExtractedFrames"
# os.makedirs(output_folder, exist_ok=True)

# # Function to extract frames at specific timesteps
# def extract_frames(video_path, timesteps, output_folder):
#     video_name = os.path.basename(video_path).split('.')[0]  # Get video name without extension
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return
    
#     fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
#     print(f"Processing {video_path} at {fps} FPS")
    
#     for timestep in timesteps:
#         frame_number = int(timestep * fps)  # Calculate frame number
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to the frame
#         success, frame = cap.read()
        
#         if success:
#             # Save the frame as an image
#             output_path = os.path.join(output_folder, f"{video_name}_t{timestep}s.jpg")
#             cv2.imwrite(output_path, frame)
#             print(f"Saved frame at {timestep}s to {output_path}")
#         else:
#             print(f"Failed to extract frame at {timestep}s for {video_path}")
    
#     cap.release()

# # Process each video
# for video_file in video_filenames:
#     extract_frames(video_file, timesteps, output_folder)

# print("Frame extraction completed.")