import torch
import matplotlib.pyplot as plt
import os
import numpy as np 

def eval_graph(tensor_datas, labels, graph_name, timestep):
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
    plt.ylabel("Value")  # Adjust the label based on your data
    plt.xlabel("Dataset")
    plt.title(f"{plot_name}")

    # Save the plot
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}_boxplot.png")
    plt.savefig(fig_path)

    plt.close()  # Close the figure



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


if __name__ == "__main__":
    main()
