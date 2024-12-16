import torch
import matplotlib.pyplot as plt
import os
import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def time_graph(tensor_datas, labels, graph_name, timestep=0.02):
    """
    Creates and saves a line graph with multiple lines from PyTorch tensors.

    Args:
        tensor_datas: A list of PyTorch tensors containing the data to plot.
        labels: A list of labels for each line.
        graph_name: The name of the graph (used for the file name).
        timestep: The timestep value (for adding to the graph title).
    """
    
    plt.figure(figsize=(10, 6))

    for tensor_data, label in zip(tensor_datas, labels):
        data_array = tensor_data.detach().cpu().numpy()
        # Create time array for x-axis
        time_array = np.arange(len(data_array)) * timestep  
        plt.plot(time_array, data_array, label=label)

    plt.title(f"{graph_name}")
    plt.xlabel("Time")  # Change x-axis label to "Time"
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle='--')
    plt.legend()  # Add legend to display labels

    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{graph_name}.png")
    plt.savefig(fig_path)  # Save the combined graph

    plt.close()  


def create_graph(y_data, x_data, labels, graph_name, label_y, label_x):
    """
    Creates and saves a line graph with multiple lines from PyTorch tensors.

    Args:
        tensor_datas: A list of PyTorch tensors containing the data to plot.
        labels: A list of labels for each line.
        graph_name: The name of the graph (used for the file name).
        timestep: The timestep value (for adding to the graph title).
    """
    
    plt.figure(figsize=(10, 6))

    for tensor_data, label in zip(y_data, labels):
        data_array = tensor_data.detach().cpu().numpy()
        # Create time array for x-axis
        x_array = x_data.detach().cpu().numpy()
        plt.plot(x_array, data_array, label=label)

    plt.title(f"{graph_name}")
    plt.xlabel(label_x)  # Change x-axis label to "Time"
    plt.ylabel(label_y)
    plt.grid(axis='y', linestyle='--')
    plt.legend()  # Add legend to display labels

    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{graph_name}.png")
    plt.savefig(fig_path)  # Save the combined graph

    plt.close()  

def polar_scatter_push_plot(radius, theta, success, plot_name):
    """
    Function to plot a polar scatter plot.
    Successful points are plotted in green, unsuccessful points in red.
    
    Args:
    - radius: A tensor containing the radius values.
    - theta: A tensor containing the angular values in radians.
    - success: A tensor containing binary values (1 for success, 0 for failure).
    """
    # Create a polar scatter plot
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')

    # Plot successful points (success == 1) in green
    ax.scatter(theta[success == 1], radius[success == 1], color='green', label='Success')

    # Plot unsuccessful points (success == 0) in red
    ax.scatter(theta[success == 0], radius[success == 0], color='red', label='Failure')

    # Add a title and legend
    ax.set_title(plot_name, va='bottom')
    ax.legend()

    # Show the plot
    #plt.show()
    # Save the plot
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}_polar_scatter.png")
    plt.savefig(fig_path)

    plt.close()  # Close the figure


def polar_boundary(radius, theta, success, ax, label=None, color=None, title=None):
    """
    Plots a decision boundary on a polar plot using SVM.

    Args:
        radius (torch.Tensor): Tensor of radius values.
        theta (torch.Tensor): Tensor of angle values.
        success (torch.Tensor): Tensor of success/failure labels.
        ax (matplotlib.axes.Axes): Axes to plot the boundary on.
        label (str, optional): Label for the current dataset (for legend). Defaults to None.
        color (str, optional): Color for the boundary line. Defaults to None.
    """

    # Convert tensors to numpy arrays
    radius = radius.numpy()
    theta = theta.numpy()
    success = success.numpy()

    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Create a DataFrame with Cartesian coordinates and tags
    df = pd.DataFrame({"x": x, "y": y, "success": success})

    # Separate features and labels
    X = df[["x", "y"]]
    y = df["success"]

    # Standardize features (optional but often helpful)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create an SVM model
    svm = SVC(kernel="rbf", C=0.9, class_weight='balanced')  # Adjust parameters as needed

    # Train the SVM model
    svm.fit(X_scaled, y)

    # Create a grid of points for decision function evaluation in Cartesian space
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Predict labels for the grid points
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rescale the grid points back to the original scale
    xx_original = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_original = yy * scaler.scale_[1] + scaler.mean_[1]

    # Convert the rescaled Cartesian grid (xx_original, yy_original) back to polar coordinates
    r = np.sqrt(xx_original**2 + yy_original**2)  # radius
    theta_boundary = np.arctan2(yy_original, xx_original)  # angle

    # Plot the decision boundary on polar axes
    ax.contour(theta_boundary, r, Z, levels=[0.5], colors=[color], linewidths=2)
    
    ax.set_title(title)
        # Set up dynamic grid lines for the polar plot
    r_max = radius.max()  # Max radius from the actual data

    #ax.set_rticks(r_ticks)  # Set radial ticks to match the data range
    ax.set_rlim(0, r_max)  # Set radial limits to cut the plot at r_max
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)  # Enable grid lines
    
def polar_scatter_boundary(radius, theta, success, ax, label=None, color=None, title=None):
    """
    Plots a decision boundary on a polar plot using SVM.

    Args:
        radius (torch.Tensor): Tensor of radius values.
        theta (torch.Tensor): Tensor of angle values.
        success (torch.Tensor): Tensor of success/failure labels.
        ax (matplotlib.axes.Axes): Axes to plot the boundary on.
        label (str, optional): Label for the current dataset (for legend). Defaults to None.
        color (str, optional): Color for the boundary line. Defaults to None.
    """

    # Convert tensors to numpy arrays
    radius = radius.numpy()
    theta = theta.numpy()
    success = success.numpy()

    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Create a DataFrame with Cartesian coordinates and tags
    df = pd.DataFrame({"x": x, "y": y, "success": success})

    # Separate features and labels
    X = df[["x", "y"]]
    y = df["success"]

    # Standardize features (optional but often helpful)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create an SVM model
    svm = SVC(kernel="rbf")  # Adjust parameters as needed

    # Train the SVM model
    svm.fit(X_scaled, y)

    # Create a grid of points for decision function evaluation in Cartesian space
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Predict labels for the grid points
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rescale the grid points back to the original scale
    xx_original = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_original = yy * scaler.scale_[1] + scaler.mean_[1]

    # Convert the rescaled Cartesian grid (xx_original, yy_original) back to polar coordinates
    r = np.sqrt(xx_original**2 + yy_original**2)  # radius
    theta_boundary = np.arctan2(yy_original, xx_original)  # angle

    # Plot the decision boundary on polar axes
    ax.contour(theta_boundary, r, Z, levels=[0.5], colors=[color], linewidths=2)
    # Plot successful points (success == 1) in green
    ax.scatter(theta[success == 1], radius[success == 1], color='green', label='Success')
    # Plot unsuccessful points (success == 0) in red
    ax.scatter(theta[success == 0], radius[success == 0], color='red', label='Failure')
    if title:
        ax.set_title(title)
    # Set up dynamic grid lines for the polar plot
    r_max = radius.max()  # Max radius from the actual data

    #ax.set_rticks(r_ticks)  # Set radial ticks to match the data range
    ax.set_rlim(0, r_max)  # Set radial limits to cut the plot at r_max
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)  # Enable grid lines


def create_multiple_box_plots(data_arrays, labels, plot_name):
    """
    Creates and saves a single plot with multiple vertical box plots from PyTorch tensors or NumPy arrays.

    Args:
        data_arrays: A list of PyTorch tensors or NumPy arrays containing the data to plot.
        labels: A list of labels corresponding to each dataset.
        plot_name: The name of the plot (used for the file name).
    """
    
    # Convert all inputs to NumPy arrays if necessary
    data_arrays = [data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data for data in data_arrays]

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Create the box plots
    plt.boxplot(data_arrays, labels=labels)

    # Add labels and title
    plt.ylabel("Tracking error")  # Adjust the label based on your data
    #plt.xlabel("Dataset")
    plt.title(f"{plot_name}")

    # Save the plot
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}_boxplot.png")
    plt.savefig(fig_path)

    plt.close()  # Close the figure

def save_tensors_to_csv(tensors, labels, filename='tensor_data.csv', directory='graphs'):
    """
    Saves multiple PyTorch tensors to a CSV file with specified labels.
    
    Parameters:
    - tensors: List of PyTorch tensors to be saved.
    - labels: List of labels corresponding to the tensors.
    - filename: Name of the CSV file to save the tensors (default is 'tensor_data.csv').
    """
    if len(tensors) != len(labels):
        raise ValueError(f"The number of tensors {len(tensors)}  must match the number of labels. {len(labels)}")
    
    tensor_dict = {
        'label': [],
        'data': [],
        'shape': [],
        'dtype': []
    }

    for tensor, label in zip(tensors, labels):
        tensor_np = tensor.numpy().flatten()
        tensor_dict['label'].append(label)
        tensor_dict['data'].append(','.join(map(str, tensor_np)))
        tensor_dict['shape'].append(tensor.shape)
        tensor_dict['dtype'].append(str(tensor.dtype))

    df = pd.DataFrame(tensor_dict)

    dir_name = os.path.join(os.getcwd(), 'outputs', directory)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, filename)
    df.to_csv(filename, index=False)

def load_tensor_from_csv(label, filename='tensor_data.csv'):
    """
    Loads a PyTorch tensor from a CSV file based on a specified label.
    
    Parameters:
    - label: Label of the tensor to be loaded.
    - filename: Name of the CSV file to load the tensor from (default is 'tensor_data.csv').
    
    Returns:
    - PyTorch tensor with the specified label.
    """

    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, filename)

    # Read the CSV file
    df = pd.read_csv(filename)
    tensor_row = df[df['label'] == label].iloc[0]

    # Get the tensor data, shape, and dtype
    tensor_data_str = tensor_row['data']
    tensor_shape = eval(tensor_row['shape'])
    
    dtype_str = tensor_row['dtype']
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
        'torch.bool': torch.bool,
        # Add more dtypes as needed
    }
    tensor_dtype = dtype_map[dtype_str]

    # Handle torch.bool tensor separately
    if tensor_dtype == torch.bool:
        # Convert the string "True,False,..." to a list of booleans
        tensor_data = [val == 'True' for val in tensor_data_str.split(',')]
    else:
        # For other types, use np.fromstring
        tensor_data = np.fromstring(tensor_data_str, sep=',')

    # Convert the data to a PyTorch tensor and reshape it
    return torch.tensor(tensor_data, dtype=tensor_dtype).reshape(tensor_shape)

def create_power_energy_bar_chart(title, names, power, energy, filename, y_lim):
    """
    Creates a bar chart with the given title, names, power, and energy values, and saves it to a file.
    Power and energy values are plotted on separate y-axes with fixed scaling.

    Parameters:
    title (str): The title of the bar chart.
    names (list): A list of names for the x-axis.
    power (torch.Tensor): A tensor of power values to plot on the primary y-axis.
    energy (torch.Tensor): A tensor of energy values to plot on the secondary y-axis.
    filename (str): The filename to save the plot.
    y_lim (tuple): A tuple specifying the y-axis limits (min, max).
    """
    # Convert PyTorch tensors to NumPy arrays
    power = power.cpu().numpy()
    energy = energy.cpu().numpy()

    # Generate x positions for the bars
    x_positions = np.arange(len(names))

    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define the colors for the bars
    power_color = 'blue'
    energy_color = 'red'

    # Plot the power values
    ax1.bar(x_positions - 0.2, power, width=0.4, color=power_color, label='Power')
    ax1.set_ylabel('Mean Power[W]', color=power_color)
    ax1.tick_params(axis='y', labelcolor=power_color)
    ax1.set_ylim((0,y_lim[0]))

    # Create the second y-axis and plot the energy values
    ax2 = ax1.twinx()
    ax2.bar(x_positions + 0.2, energy, width=0.4, color=energy_color, label='Energy')
    ax2.set_ylabel('Energy overall[Wh]', color=energy_color)
    ax2.tick_params(axis='y', labelcolor=energy_color)
    ax2.set_ylim((0,y_lim[1]))

    # Add title and labels
    ax1.set_title(title)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(names)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

def create_bar_chart(title, names, values, plot_name, y_label):
    """
    Creates and saves a bar chart.
    
    Args:
    - title (str): The title of the chart.
    - names (list): The names or labels for the bars.
    - values (torch.Tensor): The values for each bar (can be a PyTorch tensor).
    - filename (str): The filename to save the figure to.
    - y_label (str): The label for the y-axis.
    """
    # Convert values tensor to numpy array if it's a torch.Tensor
    values = values.cpu().numpy()
    # Get colormap
    cmap = plt.get_cmap('viridis')
    
    # Generate colors from the colormap
    num_bars = len(names)
    colors = cmap(np.linspace(0, 1, num_bars))

    # Create the figure and axis
    fig = plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=colors)

    # Add title and y-axis label
    plt.title(title)
    plt.ylabel(y_label)

    # Save the plot to the provided filename
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}.png")
    plt.savefig(fig_path)  # Save the xy position plot

    plt.close()

def plot_xy_position(tensor_xy, plot_name):
    """
    Creates and saves an xy position plot from a PyTorch tensor.

    Args:
        tensor_xy: A PyTorch tensor containing the xy position data.
        plot_name: The name of the plot (used for the file name).
    """
    # Convert PyTorch tensor to NumPy array, ensuring it is on the CPU
    xy_data = tensor_xy.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(xy_data[:, 0], xy_data[:, 1], marker='o', linestyle='-', color='b')
    plt.title(plot_name)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)

    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}.png")
    plt.savefig(fig_path)  # Save the xy position plot

    plt.close()

# def create_polar_plot(r, label, scale_name, plot_name):
#     theta = np.deg2rad(np.arange(0, 361, 45))

#     fig = plt.figure(dpi=200)
#     ax = fig.add_subplot(projection='polar')

#     # Iterate over each data point in r
#     if r.dim() == 1:
#         plt.polar(theta, np.append(r, r[0]), marker='o', label=label)
#         #plt.legend()
#     if r.dim() >1:
#         for i in range(r.shape[0]):
#             # Plotting the polar coordinates on the system
#             plt.polar(theta, np.append(r[i,:], r[i,0]), marker='o', label=label[i])
#             #plt.legend()


#     # Position the legend outside the plot
#     plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1.1), borderaxespad=0.)
#     # Set the radial limits
#     ax.set_rorigin(0)
#     ax.set_ylim(0, r.max())

#     # to control how far the scale is from the plot (axes coordinates)
#     def add_scale(ax, X_OFF, Y_OFF):
#         # add extra axes for the scale
#         X_OFFSET = X_OFF
#         Y_OFFSET = Y_OFF
#         rect = ax.get_position()
#         rect = (rect.xmin-X_OFFSET, rect.ymin+rect.height/2-Y_OFFSET, # x, y
#                 rect.width, rect.height/2) # width, height
#         scale_ax = ax.figure.add_axes(rect)
#         for loc in ['right', 'top', 'bottom']:
#             scale_ax.spines[loc].set_visible(False)
#         scale_ax.tick_params(bottom=False, labelbottom=False)
#         scale_ax.patch.set_visible(False) # hide white background
#         # adjust the scale
#         scale_ax.spines['left'].set_bounds(*ax.get_ylim())
#         scale_ax.set_ylim(ax.get_rorigin(), ax.get_rmax())
#         # add label to the axis
#         scale_ax.set_ylabel(scale_name)

#     add_scale(ax, 0.1 ,0.2)

#     dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     fig_path = os.path.join(dir_name, f"{plot_name}.png")
#     plt.savefig(fig_path)  # Save the xy position plot

#     plt.close()


def create_polar_plot(r, label, scale_name, plot_name, threshold=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    theta = np.deg2rad(np.arange(0, 361, 45))

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection='polar')

    # Iterate over each data point in r
    if r.ndim == 1:
        plt.polar(theta, np.append(r, r[0]), marker='o', label=label)
    elif r.ndim > 1:
        for i in range(r.shape[0]):
            plt.polar(theta, np.append(r[i, :], r[i, 0]), marker='o', label=label[i])

    # Add threshold boundary if provided
    if threshold is not None:
        boundary_theta = np.linspace(0, 2 * np.pi, 500)  # 500 points for a smooth circle
        boundary_r = np.full_like(boundary_theta, threshold)
        ax.plot(boundary_theta, boundary_r, color='red', linestyle='--', linewidth=3.5, label=f"Command = {threshold}")

    # Position the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1.1), borderaxespad=0.)
    # Set the radial limits
    ax.set_rorigin(0)
    ax.set_ylim(0, r.max())

    # Function to add scale
    def add_scale(ax, X_OFF, Y_OFF):
        X_OFFSET = X_OFF
        Y_OFFSET = Y_OFF
        rect = ax.get_position()
        rect = (rect.xmin - X_OFFSET, rect.ymin + rect.height / 2 - Y_OFFSET,  # x, y
                rect.width, rect.height / 2)  # width, height
        scale_ax = ax.figure.add_axes(rect)
        for loc in ['right', 'top', 'bottom']:
            scale_ax.spines[loc].set_visible(False)
        scale_ax.tick_params(bottom=False, labelbottom=False)
        scale_ax.patch.set_visible(False)  # hide white background
        scale_ax.spines['left'].set_bounds(*ax.get_ylim())
        scale_ax.set_ylim(ax.get_rorigin(), ax.get_rmax())
        scale_ax.set_ylabel(scale_name)

    add_scale(ax, 0.1, 0.2)

    # Create output directory if it doesn't exist
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}.png")
    plt.savefig(fig_path)  # Save the plot
    plt.close()


def main():
    data1 = torch.randn(100)
    data2 = torch.randn(100) + 1  # Shift data2 for visual differentiation
    labels = ["Data 1", "Data 2"]  # Labels for each line

    eval_graph([data1, data2], labels, "multiple_lines", 2)

    # Create PyTorch tensors with different distributions
    data1 = torch.randn(100)
    data2 = torch.rand(80) * 5 + 2
    data3 = torch.randint(0, 10, (120,))

    # Create multiple box plots in one graph
    box_labels = ["Standard Normal", "Uniform (2 to 7)", "Integers (0 to 9)"]
    create_multiple_box_plots([data1, data2, data3], box_labels, "Combined Box Plots")

    # Example tensors
    # Create a sample tensor
    # Create sample tensors
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(2, 4)
    tensor3 = torch.randn(5, 2)
    print(f"Tensor 1 shape: {tensor1.shape}")
    labels = ["tensor1", "tensor2", "tensor3"]

    # Save the tensors to a CSV file
    save_tensors_to_csv([tensor1, tensor2, tensor3], labels)

    # Load the tensors from the CSV file
    loaded_tensor1 = load_tensor_from_csv("tensor1")
    loaded_tensor2 = load_tensor_from_csv("tensor2")
    loaded_tensor3 = load_tensor_from_csv("tensor3")

    # Verify that the loaded tensors match the original tensors
    print("Original Tensor 1:")
    print(tensor1)
    print("\nLoaded Tensor 1:")
    print(loaded_tensor1)

    print("\nOriginal Tensor 2:")
    print(tensor2)
    print("\nLoaded Tensor 2:")
    print(loaded_tensor2)

    print("\nOriginal Tensor 3:")
    print(tensor3)
    print("\nLoaded Tensor 3:")
    print(loaded_tensor3)

    assert torch.equal(tensor1, loaded_tensor1), "Loaded tensor1 does not match the original tensor1."
    assert torch.equal(tensor2, loaded_tensor2), "Loaded tensor2 does not match the original tensor2."
    assert torch.equal(tensor3, loaded_tensor3), "Loaded tensor3 does not match the original tensor3."

    # print("\nAll loaded tensors match the original tensors.")

    title = "Power and Energy Bar Chart"
    names = ["Position based control"]
    power = torch.tensor([20])
    energy = torch.tensor([10])
    filename = "power_energy_bar_chart.png"
    y_lim = (30, 20)  # Set the y-axis limits
    create_power_energy_bar_chart(title, names, power, energy, filename, y_lim)


    # Example usage of xy plane
    xy_tensor = torch.tensor([[0, 0], [1, 2], [2, 3], [1, -1]], device='cuda')
    plot_name = "XY Position Plot"
    plot_xy_position(xy_tensor, plot_name)

    # Generating the X and Y axis data points
    r = torch.tensor([[10, 8, 8, 8, 8, 8, 8, 8], 
                [5, 6, 7, 8, 9, 10, 9, 8], 
                [0, 2, 3, 4, 5, 6, 5, 4]])
    label = 'Speed(m/s)'
    # Creating the polar plot using the function
    create_polar_plot(r, label, 'polar_plot')

if __name__ == "__main__":
    main()
