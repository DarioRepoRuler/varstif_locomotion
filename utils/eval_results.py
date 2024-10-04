from graphs_gen import *


# ----------------------- Evaluation of experiments: Heading directions -----------------------
speeds_comp={
    'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
    #'VIC2 w ar feet air': load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv'),
    'VIC2 w ar default': load_tensor_from_csv('local_v',filename='results_vic2.csv'),
    #'VIC2 wo ar': load_tensor_from_csv('local_v',filename='results_vic2_woar.csv'),
    #'VIC2 wo ar& w jt': load_tensor_from_csv('local_v',filename='results_vic2_jt.csv'),
    #'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_continue.csv'),
    #'VIC2 wo ar & w jt 0.05': load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv'),
    'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
    'VIC2 wo ar & w jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv')
}

vic_speeds_comp={
    'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
    'VIC1 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic1_jt.csv'),
    # 'VIC1 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic1_jt_hard.csv'),
    'VIC2 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
    #'VIC2 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv'),
    #'VIC2 best set': load_tensor_from_csv('local_v',filename='results_vic2_best_set.csv'),
    'VIC3 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic3_jt.csv'),
    # 'VIC3 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic3_jt_hard.csv'),
    
    'VIC4 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic4_jt.csv'),
    #'VIC2 bugfix': load_tensor_from_csv('local_v',filename='results_vic2_bf.csv')
}

success_rates = {
    'Baseline': load_tensor_from_csv('success_rate',filename='results_pos.csv'),
    #'VIC2 w ar feet air': load_tensor_from_csv('success_rate',filename='results_vic2_feet_air.csv'),
    #'VIC2 w ar default': load_tensor_from_csv('success_rate',filename='results_vic2.csv'),
    #'VIC2 wo ar': load_tensor_from_csv('success_rate',filename='results_vic2_woar.csv'),
    #'VIC2 wo ar& w jt': load_tensor_from_csv('success_rate',filename='results_vic2_jt.csv'),
    'VIC wo ar & w jt hard': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_continue.csv'),
    #'VIC wo ar & w jt middle': load_tensor_from_csv('success_rate', filename='results_vic2_jt_middle.csv'),
    'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_newnew.csv'),
    #'VIC2 wo ar & w jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic2_jt_harder.csv')
}

vic_success_rates = {
    'Baseline': load_tensor_from_csv('success_rate',filename='results_pos.csv'),
    'VIC1 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic1_jt.csv'),
    'VIC1 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic1_jt_hard.csv'),
    'VIC2 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_newnew.csv'),
    'VIC2 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic2_jt_harder.csv'),
    #'VIC3 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic3_jt_hard.csv'),
    'VIC4 jt 0.1': load_tensor_from_csv('success_rate',filename='results_vic4_jt.csv'),
}

speeds_t=torch.stack([speeds_comp[key] for key in speeds_comp.keys()],dim=0)
speeds_t_vic=torch.stack([vic_speeds_comp[key] for key in vic_speeds_comp.keys()],dim=0)
success_rates_t = torch.stack([success_rates[key] for key in success_rates.keys()],dim=0)
success_rates_t_vic = torch.stack([vic_success_rates[key] for key in vic_success_rates.keys()],dim=0)

create_polar_plot( speeds_t, [key for key in speeds_comp.keys()] ,'Speed (m/s)','speed_comparison')
create_polar_plot( speeds_t_vic, [key for key in vic_speeds_comp.keys()] ,'Speed (m/s)','speed_comparison_vic')
create_polar_plot( success_rates_t_vic,[key for key in vic_success_rates.keys()] ,'Success Rate','success_rate_comparison_vic')
create_polar_plot( success_rates_t,[key for key in success_rates.keys()] ,'Success Rate','success_rate_comparison')

# ----------------------- Evaluation of Energy -----------------------
cot_comp={
    'Baseline': load_tensor_from_csv('COT',filename='results_pos.csv'),
    'VIC1 jt 0.1': load_tensor_from_csv('COT',filename='results_vic1_jt.csv'),
    'VIC1 jt 0.2': load_tensor_from_csv('COT',filename='results_vic1_jt_hard.csv'),
    'VIC2 jt 0.1': load_tensor_from_csv('COT',filename='results_vic2_jt_hard_newnew.csv'),
    'VIC2 jt 0.2': load_tensor_from_csv('COT',filename='results_vic2_jt_harder.csv'),
    'VIC3 jt 0.1': load_tensor_from_csv('COT',filename='results_vic3_jt.csv'),
    'VIC4 jt 0.1': load_tensor_from_csv('COT',filename='results_vic4_jt.csv'),
}


print(f"Cost of Transport: {[cot_comp[key][0] for key in cot_comp.keys()]}")
print(f"Cost of Transport: {torch.hstack([cot_comp[key][0] for key in cot_comp.keys()]) }")
print(f"Keys: {[key for key in cot_comp.keys()]}")
cot_comp_t=torch.hstack([cot_comp[key][0] for key in cot_comp.keys()])
create_bar_chart( 'COT comparison',[key for key in cot_comp.keys()], cot_comp_t, 'COT_comparison', 'Cost of Transport')
# ----------------------- Evaluation of force push -----------------------
push_data_pos={
    
    'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename='force_push_results_rando_all1.csv')[:,0,:],
    'kick_theta': load_tensor_from_csv('kick_theta', filename='force_push_results_rando_all1.csv')[:,0,:],
    'success': load_tensor_from_csv('success_rate', filename='force_push_results_rando_all1.csv'),
}
push_data_vic2={
    'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename='force_push_results_test_vic2_jt_harder.csv')[:,0,:],
    'kick_theta': load_tensor_from_csv('kick_theta', filename='force_push_results_test_vic2_jt_harder.csv')[:,0,:],
    'success': load_tensor_from_csv('success_rate', filename='force_push_results_test_vic2_jt_harder.csv'),
}

push_data_vic3={
    'kick_force_magnitude': load_tensor_from_csv('kick_force_magnitude', filename='force_push_results_test_vic3_jt.csv')[:,0,:],
    'kick_theta': load_tensor_from_csv('kick_theta', filename='force_push_results_test_vic3_jt.csv')[:,0,:],
    'success': load_tensor_from_csv('success_rate', filename='force_push_results_test_vic3_jt.csv'),
}


#print(f"Success shape: {push_data_pos['success'].shape} and kick_theta shape: {push_data_pos['kick_theta'].shape} and kick_force_magnitude shape: {push_data_pos['kick_force_magnitude'].shape}")
polar_scatter_push_plot(push_data_pos['kick_force_magnitude'], push_data_pos['kick_theta'], push_data_pos['success'], 'Force Push Positionbased')
polar_scatter_push_plot(push_data_vic2['kick_force_magnitude'], push_data_vic2['kick_theta'], push_data_vic2['success'], 'Force Push VIC2')
polar_scatter_push_plot(push_data_vic3['kick_force_magnitude'], push_data_vic3['kick_theta'], push_data_vic3['success'], 'Force Push VIC3')


# -----------------------Evaluation of pyramid excape-----------------------
pyramid_success = {
    'Baseline': load_tensor_from_csv('success_rate',filename='pyramid_results_rando_all1.csv'),
}
success_rates_t_pyramid = torch.stack([pyramid_success[key] for key in pyramid_success.keys()],dim=0)
staircase_heights = torch.tensor([5, 6.25, 7.5, 8.75])

create_graph(success_rates_t_pyramid, staircase_heights,[key for key in pyramid_success.keys()], 'Success Rate', 'success rate', 'stair height')