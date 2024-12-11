from graphs_gen import *
import os
import matplotlib.patches as mpatches

# ----------------------- Evaluation of heading + energy -----------------------
def eval_heading(filenames, labels = None, threshold=None, file_outputname = "heading"):
    print(f"Eval heading: {filenames}")
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
        #print(f"Local v error: {torch.abs(threshold-load_tensor_from_csv('local_v',filename=filename))}")
        print(f"Local v error: {torch.mean(torch.abs(threshold-load_tensor_from_csv('local_v',filename=filename)))}")
        heading_data['success_rate'].append(load_tensor_from_csv('success_rate',filename=filename))

    heading_data['local_v'] = torch.stack(heading_data['local_v'],dim=0)
    heading_data['success_rate'] = torch.stack(heading_data['success_rate'],dim=0)
    heading_data['COT'] = torch.stack(heading_data['COT'],dim=0)
    heading_data['power'] = torch.stack(heading_data['power'],dim=0)

    create_polar_plot(heading_data['COT'], heading_data['name'], 'Cost of Transport', f'{file_outputname}_COT_compare')
    create_polar_plot(heading_data['power'], heading_data['name'], 'Power (W)', f'{file_outputname}_power_compare')
    create_polar_plot(heading_data['local_v'], heading_data['name'], 'Speed (m/s)', f'{file_outputname}_compare', threshold)
    create_polar_plot(heading_data['success_rate'], heading_data['name'], 'Success Rate', f'{file_outputname}_sr_compare')


# ----------------------- Evaluation of force push -----------------------
def eval_force_push(filenames):
    for filename in filenames:
        print(f"Eval force push: {filename}")
        push_data={
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename),
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename),
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }

        if len(push_data['kick_force_magnitude'].shape) == 3:
            push_data['kick_force_magnitude'] = push_data['kick_force_magnitude'][:,0,:]
            push_data['kick_theta'] = push_data['kick_theta'][:,0,:]
            plot_name = filename.split('.')[0]
            polar_scatter_push_plot(push_data['kick_force_magnitude'], push_data['kick_theta'], push_data['success'], plot_name)
        else: 
            print(f"Skipping {filename} experiment failed!")
            
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

# -----------------------Evaluation for rando cmd -----------------------
def eval_cmd_rando(filenames):
    for filename in filenames:
        print(f"Eval cmd rando: {filename}")
        cmd_data={
            'cmd_norm': load_tensor_from_csv('cmd_norm', filename=filename),
            'success': load_tensor_from_csv('success', filename=filename)[:,0,:],
            'cmd_theta': load_tensor_from_csv('cmd_theta', filename=filename),
        }
        indices = torch.where(torch.logical_and(cmd_data['cmd_norm'][:,:,0] > 0.01, cmd_data['cmd_norm'][:,:,0] < 1.5))
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
    #plt.show()




def main():

    # Get the results from the output directory
    output_dir = os.path.join(os.getcwd(), 'outputs', 'graphs')
    if not os.path.exists(output_dir):
        assert False, f"Output directory does not exist: {output_dir}"

    files= os.listdir(output_dir)
    filenames = {'cmd_rando':[], 'force_push':[], 'heading_directions':[], 'pyramid':[]}
    
    # Best run names of each control
    vic_run_names = {'VIC1': '2024-11-25/11-06-34', 'VIC2': '2024-11-25/10-20-58', 'VIC3': '2024-11-25/14-07-08', 'VIC4': '2024-12-02/11-50-31'}
    pos_run_names = {'P20': '2024-11-25/14-03-45', 'P50': '2024-11-24/09-58-49'}

    # Replace / in the filenames
    for key in vic_run_names:
        vic_run_names[key] = vic_run_names[key].replace('/', '_')
    for key in pos_run_names:
        pos_run_names[key] = pos_run_names[key].replace('/', '_')
    
    # ----------------------- Evaluation of heading -----------------------
    ## Different speeds
    filenames_headings= [f'heading_directions_results_{pos_run_names['P20']}_lowspeed.csv', f'heading_directions_results_{pos_run_names['P50']}_lowspeed.csv' , f'heading_directions_results_{vic_run_names['VIC2']}_lowspeed.csv']
    labels = ['P20', 'P50','PLS']
    eval_heading(filenames_headings, labels= labels, threshold=0.5, file_outputname = "heading_lowspeed")

    filenames_headings = [f'heading_directions_results_{pos_run_names["P20"]}_midspeed.csv', f'heading_directions_results_{pos_run_names["P50"]}_midspeed.csv' , f'heading_directions_results_{vic_run_names["VIC2"]}_midspeed.csv']
    labels = ['P20', 'P50','PLS']
    eval_heading(filenames_headings, labels= labels, threshold=1.0, file_outputname = "heading_midspeed")

    filenames_headings = [f'heading_directions_results_{pos_run_names["P20"]}_highspeed.csv', f'heading_directions_results_{pos_run_names["P50"]}_highspeed.csv' , f'heading_directions_results_{vic_run_names["VIC2"]}_highspeed.csv']
    labels = ['P20', 'P50','PLS']
    eval_heading(filenames_headings, labels= labels, threshold=1.3, file_outputname = "heading_highspeed")

    
    # ----------------------- Evaluation of Payload -----------------------
    # filenames_headings= ['heading_directions_test_payload_2024-11-25_14-03-45.csv', 'heading_directions_test_payload_2024-11-24_09-58-49.csv' , 'heading_directions_test_payload_2024-11-25_10-20-58.csv']
    # labels = ['P20', 'P50','PLS']
    # eval_heading(filenames_headings, labels= labels, threshold=0.3, file_outputname = "heading_payload")

    # filenames_headings= ['heading_directions_test_2024-11-25_11-06-34.csv','heading_directions_test_2024-11-25_10-20-58.csv', 'heading_directions_test_2024-11-25_14-07-08.csv','heading_directions_test_2024-12-02_11-50-31.csv'] #'heading_directions_results_2024-11-25_14-07-08.csv',
    # labels = ['PJS', 'PLS', 'IJS' ,'HJLS'] # vic 1, 2,3,4
    # eval_heading(filenames_headings, labels= labels, threshold=0.8, file_outputname = "heading vic comparison")
    


    # ----------------------- Evaluation of force push -----------------------
    #eval_force_push_boundary(['force_push_test_automation_2024-11-25_14-03-45.csv','force_push_test_automation_2024-11-25_10-20-58.csv'], labels = ['Baseline', 'VIC2'])
    compare_runs =[ f'force_push_test_automation_{pos_run_names["P20"]}.csv', f'force_push_test_automation_{vic_run_names["VIC2"]}.csv', f'force_push_test_automation_{pos_run_names["P50"]}.csv']
    labels = ['P20',  'PLS','P50']
    eval_force_push_scatter_boundary(compare_runs, labels = labels)
    eval_force_push_boundary(compare_runs, labels = labels)
    compare_runs =[f'force_push_test_automation_{vic_run_names["VIC1"]}.csv', f'force_push_test_automation_{vic_run_names["VIC2"]}.csv', f'force_push_test_automation_{vic_run_names["VIC3"]}.csv', f'force_push_test_automation_{vic_run_names["VIC4"]}.csv']
    labels = ['PJS', 'PLS', 'IJS' ,'HJLS']
    eval_force_push_scatter_boundary(compare_runs, labels = labels)
    eval_force_push_boundary(compare_runs, labels = labels)


    # ----------------------- Evaluation of command random -----------------------
    #filenames = ['cmd_rando_xy_test_vic2_0810_1.csv', 'cmd_rando_xy_model_1500.csv', 'cmd_rando_xy_11-32-08.csv', 'cmd_rando_xy_13-00-31.csv','cmd_rando_xy_11-32-08_1500.csv', 'cmd_rando_xy_11-32-08.csv']
    #eval_cmd_rando(filenames['cmd_rando'])



if __name__ == '__main__':
    main()