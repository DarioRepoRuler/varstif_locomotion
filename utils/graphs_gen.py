import torch
import matplotlib.pyplot as plt
import os
import numpy as np 
import pandas as pd

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

def save_tensors_to_csv(tensors, labels, filename='tensor_data.csv'):
    """
    Saves multiple PyTorch tensors to a CSV file with specified labels.
    
    Parameters:
    - tensors: List of PyTorch tensors to be saved.
    - labels: List of labels corresponding to the tensors.
    - filename: Name of the CSV file to save the tensors (default is 'tensor_data.csv').
    """
    if len(tensors) != len(labels):
        raise ValueError("The number of tensors must match the number of labels.")
    
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

    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
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

    df = pd.read_csv(filename)
    tensor_row = df[df['label'] == label].iloc[0]
    tensor_data = np.fromstring(tensor_row['data'], sep=',')
    tensor_shape = eval(tensor_row['shape'])
    
    dtype_str = tensor_row['dtype']
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
        # Add more dtypes as needed
    }
    tensor_dtype = dtype_map[dtype_str]

    return torch.tensor(tensor_data, dtype=tensor_dtype).reshape(tensor_shape)



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

    print("\nAll loaded tensors match the original tensors.")

if __name__ == "__main__":
    main()
