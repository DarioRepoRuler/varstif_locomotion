from graphs_gen import *
import os
import matplotlib.patches as mpatches

# ----------------------- Evaluation of heading -----------------------
def eval_heading(filenames, labels = None, threshold=None, file_outputname = "heading"):
    #print(f"Eval heading: {filenames}")
    heading_data = {'name':[],'local_v':[], 'success_rate':[], 'COT':[], 'power':[]}
    for i,filename in enumerate(filenames):
        if labels == None:
            label = filename.split('.')[0]
            label = label.split('_')[-1]
            label = f'model_{label}'
        else:
            label = labels[i]
        heading_data['name'].append(label)
        heading_data['local_v'].append(load_tensor_from_csv('local_v',filename=filename))
        heading_data['COT'].append(load_tensor_from_csv('COT',filename=filename))
        heading_data['power'].append(load_tensor_from_csv('power',filename=filename))
        heading_data['success_rate'].append(load_tensor_from_csv('success_rate',filename=filename))

    heading_data['local_v'] = torch.stack(heading_data['local_v'],dim=0)
    heading_data['success_rate'] = torch.stack(heading_data['success_rate'],dim=0)
    heading_data['COT'] = torch.stack(heading_data['COT'],dim=0)
    heading_data['power'] = torch.stack(heading_data['power'],dim=0)

    create_polar_plot(heading_data['COT'], heading_data['name'], 'Cost of Transport', f'{file_outputname}_COT_compare')
    create_polar_plot(heading_data['power'], heading_data['name'], 'Power (W)', f'{file_outputname}_power_compare')
    create_polar_plot(heading_data['local_v'], heading_data['name'], 'Speed (m/s)', f'{file_outputname}_compare', threshold)
    create_polar_plot(heading_data['success_rate'], heading_data['name'], 'Success Rate', f'{file_outputname}_sr_compare')


# ----------------------- Evaluation of Energy -----------------------
def eval_cot_heading(filenames, labels = None):
    cot_data = {'name':[],'COT':[]}
    print(f"Lables: {labels}")
    for i,filename in enumerate(filenames):
        if labels == None:
            label = filename.split('.')[0]
            label = label.split('_')[-1]
            label = f'model_{label}'
        else: 
            label = labels[i]
        cot_data['name'].append(label)
        #cot_data['COT'].append(torch.mean(load_tensor_from_csv('COT',filename=filename)))
        cot_data['COT'].append(load_tensor_from_csv('COT',filename=filename)[0])
    #print(f"COT: {cot_data['COT']}")
    cot_data['COT'] = torch.stack(cot_data['COT'],dim=0)
    create_bar_chart('COT comparison', cot_data['name'], cot_data['COT'], 'COT_comparison', 'Cost of Transport')


# ----------------------- Evaluation of force push -----------------------

def eval_force_push(filenames):
    for filename in filenames:
        print(f"Eval force push: {filename}")
        push_data={
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename),
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename),
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }
        print(f"Shape of push data: {push_data['kick_force_magnitude'].shape} shape of success: {push_data['success'].shape}")
        if len(push_data['kick_force_magnitude'].shape) == 3:
            push_data['kick_force_magnitude'] = push_data['kick_force_magnitude'][:,0,:]
            push_data['kick_theta'] = push_data['kick_theta'][:,0,:]
        plot_name = filename.split('.')[0]    
        polar_scatter_push_plot(push_data['kick_force_magnitude'].flatten(), push_data['kick_theta'].flatten(), push_data['success'].flatten(), plot_name)
            
def eval_force_push_scatter_boundary(filenames, labels=None):
    """
    Create independent polar scatter boundary plots for each file.

    Parameters:
    - filenames: str or list of str, paths to the data files.
    - labels: list of str, labels for the plots (optional).
    """
    # Ensure filenames is a list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Prepare output directory
    output_dir = os.path.join(os.getcwd(), 'outputs', 'graphs')
    os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(filenames):
        print(f"Evaluating force push (with boundary): {filename}")

        # Load data
        push_data = {
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename)[:, 0, :],
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename)[:, 0, :],
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }

        # Create a polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Scatter plot with the fixed color 'blue'
        polar_scatter_boundary(
            push_data['kick_force_magnitude'].flatten(),
            push_data['kick_theta'].flatten(),
            push_data['success'].flatten(),
            ax=ax,
            label=labels[i] if labels else None,  # Use label if provided
            color='blue',  # Always use blue
            title=f'{labels[i]} Force push results'
        )

        # Add legend only for the current plot
        # if labels:
        #     ax.legend([mpatches.Patch(color='blue', label=labels[i])])

        # Save the plot
        fig_path = os.path.join(output_dir, f"{labels[i] if labels else f'plot_{i}'}_scatter_boundary.png")
        plt.savefig(fig_path)
        plt.close(fig)  # Close the figure to free memory


# -----------------------Evaluation of pyramid excape-----------------------
# pyramid_success = {
#     'Baseline': load_tensor_from_csv('success_rate',filename='pyramid_results_rando_all1.csv'),
# }
# success_rates_t_pyramid = torch.stack([pyramid_success[key] for key in pyramid_success.keys()],dim=0)
# staircase_heights = torch.tensor([5, 6.25, 7.5, 8.75])

#create_graph(success_rates_t_pyramid, staircase_heights,[key for key in pyramid_success.keys()], 'Success Rate', 'success rate', 'stair height')


# -----------------------Evaluation for rando cmd -----------------------
def eval_cmd_rando(filenames):
    for filename in filenames:
        print(f"Eval cmd rando: {filename}")
        cmd_data={
            'cmd_norm': load_tensor_from_csv('cmd_norm', filename=filename),
            'success': load_tensor_from_csv('success', filename=filename)[:,0,:],
            'cmd_theta': load_tensor_from_csv('cmd_theta', filename=filename),
        }
        #print(f"Preprocessed data: {cmd_data['cmd_norm'][:,:,0]}")
        indices = torch.where(torch.logical_and(cmd_data['cmd_norm'][:,:,0] > 0.01, cmd_data['cmd_norm'][:,:,0] < 1.5))
        #print(f"Indices: {indices}")
        #print(f"Shape of cmd_norm: {cmd_data['cmd_norm'][:,:,0].shape} and theta: {cmd_data['cmd_theta'].shape} and success: {cmd_data['success'].shape}")
        plot_name = filename.split('.')[0]
        polar_scatter_push_plot(cmd_data['cmd_norm'][indices[0], indices[1],0], cmd_data['cmd_theta'][indices[0],indices[1]], cmd_data['success'][indices[0], indices[1]], plot_name)


def eval_force_push_boundary(filenames, labels=None):
    # Check if filenames is a list
    if not isinstance(filenames, list):
        filenames = [filenames]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # Create a polar plot

    # Define a list of colors for different boundaries
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    patches = []
    for i,label in enumerate(labels):
        patches.append(mpatches.Patch(color=colors[i], label=label))

    for i, filename in enumerate(filenames):
        print(f"Eval force push(with boundary): {filename}")
        push_data = {
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename)[:, 0, :],
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename)[:, 0, :],
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }
        # print(f"Shape of push data: {push_data['kick_force_magnitude'].shape} shape of success: {push_data['success'].shape}")
        threshold = 200.0
        print(f"Success rate: {torch.sum(push_data['success'][push_data['kick_force_magnitude']<=threshold])/torch.sum(push_data['kick_force_magnitude']<=threshold)}")
        label = labels[i]  # Extract label from filename (optional)
        color = colors[i % len(colors)]  # Cycle through the colors

        # Pass the polar axes, label, and color to the function
        polar_boundary(push_data['kick_force_magnitude'].flatten(),
                                         push_data['kick_theta'].flatten(),
                                         push_data['success'].flatten(),
                                         ax, label=label, color=color, title='Force push comparison')
        
    # Add legend to the plot with the provided labels
    ax.legend(handles=patches)
    # Save the combined plot
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    plot_name = os.path.join(dir_name, f'force_push_comparison.png')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}_polar_scatter.png")
    plt.savefig(fig_path)
    #plt.show()

def eval_cmd_rando_boundary(filenames, labels=None):
    # Check if filenames is a list
    if not isinstance(filenames, list):
        filenames = [filenames]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # Create a polar plot

    # Define a list of colors for different boundaries
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    patches = []
    for i,label in enumerate(labels):
        patches.append(mpatches.Patch(color=colors[i], label=label))

    for i, filename in enumerate(filenames):
        print(f"Eval cmd rando(with boundary): {filename}")
        cmd_data={
            'cmd_norm': load_tensor_from_csv('cmd_norm', filename=filename),
            'success': load_tensor_from_csv('success', filename=filename)[:,0,:],
            'cmd_theta': load_tensor_from_csv('cmd_theta', filename=filename),
        }
        
        label = labels[i]  # Extract label from filename (optional)
        color = colors[i % len(colors)]  # Cycle through the colors

        # Pass the polar axes, label, and color to the function
        polar_boundary(cmd_data['cmd_norm'].flatten(),
                                         cmd_data['cmd_theta'].flatten(),
                                         cmd_data['success'].flatten(),
                                         ax, label=label, color=color, title='Command comparison')
        
    # Add legend to the plot with the provided labels
    ax.legend(handles=patches)

    # Save the combined plot
    dir_name = os.path.join(os.getcwd(), 'outputs', 'graphs')
    plot_name = os.path.join(dir_name, f'cmd_rando_comparison.png')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_path = os.path.join(dir_name, f"{plot_name}_polar_scatter.png")
    plt.savefig(fig_path)




def main():

    # Get the results from the output directory
    output_dir = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(output_dir):
        assert False, f"Output directory does not exist: {output_dir}"
    files= os.listdir(output_dir)
    filenames = {'cmd_rando':[], 'force_push':[], 'heading_directions':[], 'pyramid':[]}
    for file in files:
        if 'cmd_rando_xy' in file and 'csv' in file:
            filenames['cmd_rando'].append(file)
        if 'force_push_test' in file and 'csv' in file:
            filenames['force_push'].append(file)
        if 'heading_directions' in file and 'csv' in file:
            filenames['heading_directions'].append(file)
    # ----------------------- Evaluation of heading -----------------------
    eval_heading(filenames['heading_directions'])

    # ----------------------- Evaluation of Energy -----------------------
    eval_cot_heading(filenames['heading_directions'])


    # ----------------------- Evaluation of force push -----------------------
    eval_force_push(filenames['force_push'])
    # ----------------------- Comparison of some runs -----------------------
    compare_runs = ['force_push_test_automation_2024-11-25_14-03-45.csv','force_push_test_automation_2024-11-25_10-20-58.csv', 'force_push_test_automation_2024-11-24_09-58-49.csv']
    labels = ['P20',  'PLS','P50']
    eval_force_push_scatter_boundary(compare_runs, labels = labels)
    eval_force_push_boundary(compare_runs, labels = labels)


    # ----------------------- Evaluation of command random -----------------------
    eval_cmd_rando_boundary(['cmd_rando_xy_13-00-31.csv', 'cmd_rando_xy_2024-10-11_13-54-32.csv'], labels = ['Baseline', 'VIC2'])
    eval_cmd_rando(filenames['cmd_rando'])



if __name__ == '__main__':
    main()