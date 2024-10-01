from graphs_gen import *


# default_speed = load_tensor_from_csv('local_v' ,filename='results_pos.csv')
# vic2_speed = load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv')
# vic2_woar_speed = load_tensor_from_csv('local_v',filename='results_vic2_woar.csv')
# vic2_default_speed = load_tensor_from_csv('local_v',filename='results_vic2.csv')
# vic2_jt_speed = load_tensor_from_csv('local_v',filename='results_vic2_jt.csv')
# vic2_jt_hard_speed = load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_continue.csv')
# vic2_jz_middle_speed = load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv')

speeds_comp={
    'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
    'VIC2 w ar feet air': load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv'),
    'VIC2 w ar default': load_tensor_from_csv('local_v',filename='results_vic2.csv'),
    'VIC2 wo ar': load_tensor_from_csv('local_v',filename='results_vic2_woar.csv'),
    'VIC2 wo ar& w jt': load_tensor_from_csv('local_v',filename='results_vic2_jt.csv'),
    'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_continue.csv'),
    'VIC2 wo ar & w jt 0.05': load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv'),
    'VIC2 wo ar & w jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
    'VIC2 wo ar & w jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv')
}

vic_speeds_comp={
    'Baseline': load_tensor_from_csv('local_v' ,filename='results_pos.csv'),
    # #'VIC1 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic1_jt.csv'),
    # 'VIC1 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic1_jt_hard.csv'),
    # #'VIC2 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv'),
    'VIC2 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic2_jt_harder.csv'),
    # # #'VIC3 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic3_jt.csv'),
    # 'VIC3 jt 0.2': load_tensor_from_csv('local_v',filename='results_vic3_jt_hard.csv'),
    'VIC2 best set': load_tensor_from_csv('local_v',filename='results_vic2_best_set.csv'),
    # 'VIC4 jt 0.1': load_tensor_from_csv('local_v',filename='results_vic4_jt.csv'),
    'VIC2 bugfix': load_tensor_from_csv('local_v',filename='results_vic2_bf.csv')
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
    'VIC3 jt 0.2': load_tensor_from_csv('success_rate',filename='results_vic3_jt_hard.csv'),
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