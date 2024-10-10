from graphs_gen import *

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



# ----------------------- Evaluation of experiments: Heading directions -----------------------
# speeds_comp={
#     'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
#     #'VIC2 w ar feet air': load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv'),
#     'VIC2 w ar default': load_tensor_from_csv('local_v',filename='results_vic2.csv'),
#     #'VIC2 wo ar': load_tensor_from_csv('local_v',filename='results_vic2_woar.csv'),
#     #'VIC2 wo ar& w jt': load_tensor_from_csv('local_v',filename='results_vic2_jt.csv'),
#     #'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_continue.csv'),
#     #'VIC2 wo ar & w jt 0.05': load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv'),
#     'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
#     'VIC2 wo ar & w jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv')
# }

# vic_speeds_comp={
#     'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
#     'VIC1 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic1_jt.csv'),
#     # 'VIC1 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic1_jt_hard.csv'),
#     'VIC2 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
#     #'VIC2 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv'),
#     #'VIC2 best set': load_tensor_from_csv('local_v',filename='results_vic2_best_set.csv'),
#     'VIC3 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic3_jt.csv'),
#     # 'VIC3 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic3_jt_hard.csv'),
    
#     'VIC4 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic4_jt.csv'),
#     #'VIC2 bugfix': load_tensor_from_csv('local_v',filename='results_vic2_bf.csv')
#     'VIC 2 0810': load_tensor_from_csv('local_v',filename='results_vic2_0810.csv')
# }

# success_rates = {
#     'Baseline': load_tensor_from_csv('success_rate',filename='results_pos.csv'),
#     #'VIC2 w ar feet air': load_tensor_from_csv('success_rate',filename='results_vic2_feet_air.csv'),
#     #'VIC2 w ar default': load_tensor_from_csv('success_rate',filename='results_vic2.csv'),
#     #'VIC2 wo ar': load_tensor_from_csv('success_rate',filename='results_vic2_woar.csv'),
#     #'VIC2 wo ar& w jt': load_tensor_from_csv('success_rate',filename='results_vic2_jt.csv'),
#     'VIC wo ar & w jt hard': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_continue.csv'),
#     #'VIC wo ar & w jt middle': load_tensor_from_csv('success_rate', filename='results_vic2_jt_middle.csv'),
#     'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_newnew.csv'),
#     #'VIC2 wo ar & w jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic2_jt_harder.csv')
     
# }

# vic_success_rates = {
#     'Baseline': load_tensor_from_csv('success_rate',filename='results_pos.csv'),
#     'VIC1 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic1_jt.csv'),
#     'VIC1 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic1_jt_hard.csv'),
#     'VIC2 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_newnew.csv'),
#     'VIC2 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic2_jt_harder.csv'),
#     #'VIC3 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic3_jt_hard.csv'),
#     'VIC4 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic4_jt.csv'),
# }

# speeds_t=torch.stack([speeds_comp[key] for key in speeds_comp.keys()],dim=0)
# speeds_t_vic=torch.stack([vic_speeds_comp[key] for key in vic_speeds_comp.keys()],dim=0)
# success_rates_t = torch.stack([success_rates[key] for key in success_rates.keys()],dim=0)
# success_rates_t_vic = torch.stack([vic_success_rates[key] for key in vic_success_rates.keys()],dim=0)

# create_polar_plot( speeds_t, [key for key in speeds_comp.keys()] ,'Speed (m/s)','speed_comparison')
# create_polar_plot( speeds_t_vic, [key for key in vic_speeds_comp.keys()] ,'Speed (m/s)','speed_comparison_vic')
# create_polar_plot( success_rates_t_vic,[key for key in vic_success_rates.keys()] ,'Success Rate','success_rate_comparison_vic')
# create_polar_plot( success_rates_t,[key for key in success_rates.keys()] ,'Success Rate','success_rate_comparison')


def eval_heading(filenames):
    heading_data = {'name':[],'local_v':[], 'success_rate':[]}
    for filename in filenames:
        heading_data['name'].append(filename)
        heading_data['local_v'].append(load_tensor_from_csv('local_v',filename=filename))
        heading_data['success_rate'].append(load_tensor_from_csv('success_rate',filename=filename))

    heading_data['local_v'] = torch.stack(heading_data['local_v'],dim=0)
    heading_data['success_rate'] = torch.stack(heading_data['success_rate'],dim=0)

    create_polar_plot(heading_data['local_v'], heading_data['name'], 'Speed (m/s)', 'heading_speed_comparison')
    create_polar_plot(heading_data['success_rate'], heading_data['name'], 'Success Rate', 'heading_success_rate_comparison')

#filenames= ['heading_directions_results_model_1500.csv', 'heading_directions_results_11-32-08.csv'] #'results_vic2_jt_hard_newnew.csv', 'results_vic3_jt.csv', 'results_vic2_0810.csv']
eval_heading(filenames['heading_directions'])


# ----------------------- Evaluation of Energy -----------------------
def eval_cot_heading(filenames):
    cot_data = {'name':[],'COT':[]}
    for filename in filenames:
        cot_data['name'].append(filename)
        cot_data['COT'].append(load_tensor_from_csv('COT',filename=filename)[0])
    print(f"COT: {cot_data['COT']}")
    cot_data['COT'] = torch.stack(cot_data['COT'],dim=0)
    create_bar_chart('COT comparison', cot_data['name'], cot_data['COT'], 'COT_comparison', 'Cost of Transport')

#filenames_heading = ['heading_directions_results_model_1500.csv', 'heading_directions_results_11-32-08.csv' ]# 'results_vic2_jt_hard_newnew.csv', 'results_vic3_jt.csv', 'results_vic2_0810.csv']
eval_cot_heading(filenames['heading_directions'])

# cot_comp={
#     'Baseline': load_tensor_from_csv('COT',filename='results_pos.csv'),
#     'VIC1 jt 0.1': load_tensor_from_csv('COT',filename='results_vic1_jt.csv'),
#     'VIC1 jt 0.2': load_tensor_from_csv('COT',filename='results_vic1_jt_hard.csv'),
#     'VIC2 jt 0.1': load_tensor_from_csv('COT',filename='results_vic2_jt_hard_newnew.csv'),
#     'VIC2 jt 0.2': load_tensor_from_csv('COT',filename='results_vic2_jt_harder.csv'),
#     'VIC3 jt 0.1': load_tensor_from_csv('COT',filename='results_vic3_jt.csv'),
#     'VIC4 jt 0.1': load_tensor_from_csv('COT',filename='results_vic4_jt.csv'),
# }


# print(f"Cost of Transport: {[cot_comp[key][0] for key in cot_comp.keys()]}")
# print(f"Cost of Transport: {torch.hstack([cot_comp[key][0] for key in cot_comp.keys()]) }")
# print(f"Keys: {[key for key in cot_comp.keys()]}")
# cot_comp_t=torch.hstack([cot_comp[key][0] for key in cot_comp.keys()])
# create_bar_chart( 'COT comparison',[key for key in cot_comp.keys()], cot_comp_t, 'COT_comparison', 'Cost of Transport')

# ----------------------- Evaluation of force push -----------------------

def eval_force_push(filenames):
    for filename in filenames:
        push_data={
            'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename=filename)[:,0,:],
            'kick_theta': load_tensor_from_csv('kick_theta', filename=filename)[:,0,:],
            'success': load_tensor_from_csv('success_rate', filename=filename),
        }
        plot_name = filename.split('.')[0]
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