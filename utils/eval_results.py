from graphs_gen import *
import os
import matplotlib.patches as mpatches

# Get the results from the output directory
output_dir = '/home/dario/Documents/TALocoMotion/outputs/graphs/'
if not os.path.exists(output_dir):
    assert False, f"Output directory does not exist: {output_dir}"
files= os.listdir(output_dir)
filenames = {'cmd_rando':[], 'force_push':[], 'heading_directions':[], 'pyramid':[]}
for file in files:
    if 'cmd_rando_xy' in file and 'csv' in file:
        filenames['cmd_rando'].append(file)
    if 'force_push' in file and 'csv' in file:
        filenames['force_push'].append(file)
    if 'heading_directions' in file and 'csv' in file:
        filenames['heading_directions'].append(file)


# ----------------------- Evaluation of heading -----------------------
def eval_heading(filenames, labels = None):
    heading_data = {'name':[],'local_v':[], 'success_rate':[]}
    for i,filename in enumerate(filenames):
        if labels == None:
            label = filename.split('.')[0]
            label = label.split('_')[-1]
            label = f'model_{label}'
        else:
            label = labels[i]
        heading_data['name'].append(label)
        heading_data['local_v'].append(load_tensor_from_csv('local_v',filename=filename))
        heading_data['success_rate'].append(load_tensor_from_csv('success_rate',filename=filename))

    heading_data['local_v'] = torch.stack(heading_data['local_v'],dim=0)
    heading_data['success_rate'] = torch.stack(heading_data['success_rate'],dim=0)

    create_polar_plot(heading_data['local_v'], heading_data['name'], 'Speed (m/s)', 'heading_speed_comparison')
    create_polar_plot(heading_data['success_rate'], heading_data['name'], 'Success Rate', 'heading_success_rate_comparison')

filenames_headings= ['heading_directions_results_13-00-31.csv' , 'heading_directions_results_2024-10-15_22-15-05.csv', 'heading_directions_results_2024-10-10_14-00-27.csv', 'heading_directions_results_2024-10-11_15-11-37.csv'] #'results_vic2_jt_hard_newnew.csv', 'results_vic3_jt.csv', 'results_vic2_0810.csv']
labels = ['Baseline', 'VIC1', 'VIC2', 'VIC4']
eval_heading(filenames_headings, labels= labels)


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

filenames_heading = ['heading_directions_results_13-00-31.csv', 'heading_directions_results_2024-10-11_13-54-32.csv', 'heading_directions_results_2024-10-10_14-00-27.csv' ]# 'results_vic2_jt_hard_newnew.csv', 'results_vic3_jt.csv', 'results_vic2_0810.csv']
labels = ['Baseline', 'VIC2 with feet contacts', 'VIC2 narrow stiff range']
#eval_cot_heading(filenames['heading_directions'])
eval_cot_heading(filenames_heading, labels)

# ----------------------- Evaluation of force push -----------------------

def eval_force_push(filenames):
    for filename in filenames:
        push_data={
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename)[:,0,:],
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename)[:,0,:],
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }
        plot_name = filename.split('.')[0]
        print(f"Filename: {filename}")
        polar_scatter_push_plot(push_data['kick_force_magnitude'], push_data['kick_theta'], push_data['success'], plot_name)

filenames_force = ['force_push_results_rando_all1.csv', 'force_push_results_test_vic2_jt_harder.csv', 'force_push_results_test_vic3_jt.csv', 'force_push_results_test_vic2_0810.csv', 'force_push_results_test_vic2_0810_1.csv']
filenames_force = ['force_push_results_model_1500.csv', 'force_push_results_test_vic2_jt_harder.csv', 'force_push_results_test_vic3_jt.csv', 'force_push_results_test_vic2_0810.csv', 'force_push_results_test_vic2_0810_1.csv']

eval_force_push(filenames['force_push'])

# -----------------------Evaluation of pyramid excape-----------------------
pyramid_success = {
    'Baseline': load_tensor_from_csv('success_rate',filename='pyramid_results_rando_all1.csv'),
}
success_rates_t_pyramid = torch.stack([pyramid_success[key] for key in pyramid_success.keys()],dim=0)
staircase_heights = torch.tensor([5, 6.25, 7.5, 8.75])

create_graph(success_rates_t_pyramid, staircase_heights,[key for key in pyramid_success.keys()], 'Success Rate', 'success rate', 'stair height')


# -----------------------Evaluation for rando cmd -----------------------
def eval_cmd_rando(filenames):
    for filename in filenames:
        cmd_data={
            'cmd_norm': load_tensor_from_csv('cmd_norm', filename=filename),
            'success': load_tensor_from_csv('success', filename=filename)[:,0,:],
            'cmd_theta': load_tensor_from_csv('cmd_theta', filename=filename),
        }
        #print(f"Preprocessed data: {cmd_data['cmd_norm'][:,:,0]}")
        indices = torch.where(torch.logical_and(cmd_data['cmd_norm'][:,:,0] > 0.2, cmd_data['cmd_norm'][:,:,0] < 1.5))
        #print(f"Indices: {indices}")
        #print(f"Shape of cmd_norm: {cmd_data['cmd_norm'][:,:,0].shape} and theta: {cmd_data['cmd_theta'].shape} and success: {cmd_data['success'].shape}")
        plot_name = filename.split('.')[0]
        polar_scatter_push_plot(cmd_data['cmd_norm'][indices[0], indices[1],0], cmd_data['cmd_theta'][indices[0],indices[1]], cmd_data['success'][indices[0], indices[1]], plot_name)



#filenames = ['cmd_rando_xy_test_vic2_0810_1.csv', 'cmd_rando_xy_model_1500.csv', 'cmd_rando_xy_11-32-08.csv', 'cmd_rando_xy_13-00-31.csv','cmd_rando_xy_11-32-08_1500.csv', 'cmd_rando_xy_11-32-08.csv']
eval_cmd_rando(filenames['cmd_rando'])


# Eval boundaries for force push and rando cmd
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
        print(f"Filename: {filename}")
        push_data = {
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename)[:, 0, :],
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename)[:, 0, :],
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }
        
        label = labels[i]  # Extract label from filename (optional)
        color = colors[i % len(colors)]  # Cycle through the colors

        # Pass the polar axes, label, and color to the function
        plot_polar_scatter_with_boundary(push_data['kick_force_magnitude'].flatten(),
                                         push_data['kick_theta'].flatten(),
                                         push_data['success'].flatten(),
                                         ax, label=label, color=color, title='Force push comparison')
        
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
        print(f"Filename: {filename}")
        cmd_data={
            'cmd_norm': load_tensor_from_csv('cmd_norm', filename=filename),
            'success': load_tensor_from_csv('success', filename=filename)[:,0,:],
            'cmd_theta': load_tensor_from_csv('cmd_theta', filename=filename),
        }
        
        label = labels[i]  # Extract label from filename (optional)
        color = colors[i % len(colors)]  # Cycle through the colors

        # Pass the polar axes, label, and color to the function
        plot_polar_scatter_with_boundary(cmd_data['cmd_norm'].flatten(),
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



# Example call
eval_force_push_boundary(['force_push_results_2024-10-09_13-00-31.csv', 'force_push_results_2024-10-11_13-54-32.csv'], labels = ['Baseline', 'VIC2'])
#eval_cmd_rando_boundary(['cmd_rando_xy_13-00-31.csv', 'cmd_rando_xy_2024-10-11_13-54-32.csv'], labels = ['Baseline', 'VIC2'])