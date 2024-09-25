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
    #'VIC2 w ar feet air': load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv'),
    #'VIC2 w ar default': load_tensor_from_csv('local_v',filename='results_vic2.csv'),
    #'VIC2 wo ar': load_tensor_from_csv('local_v',filename='results_vic2_woar.csv'),
    'VIC2 wo ar& w jt': load_tensor_from_csv('local_v',filename='results_vic2_jt.csv'),
    'VIC2 wo ar & w jt hard': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_continue.csv'),
    'VIC2 wo ar & w jt middle': load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv'),
    'VIC2 wo ar & w jt hard new': load_tensor_from_csv('local_v',filename='results_vic2_jt_hard_newnew.csv')
}

success_rates = {
    'Baseline': load_tensor_from_csv('success_rate',filename='results_pos.csv'),
    'VIC2 w ar feet air': load_tensor_from_csv('success_rate',filename='results_vic2_feet_air.csv'),
    'VIC2 w ar default': load_tensor_from_csv('success_rate',filename='results_vic2.csv'),
    'VIC2 wo ar': load_tensor_from_csv('success_rate',filename='results_vic2_woar.csv'),
    'VIC2 wo ar& w jt': load_tensor_from_csv('success_rate',filename='results_vic2_jt.csv'),
    'VIC wo ar & w jt hard': load_tensor_from_csv('success_rate',filename='results_vic2_jt_hard_continue.csv'),
    'VIC wo ar & w jt middle': load_tensor_from_csv('success_rate', filename='results_vic2_jt_middle.csv')
}




speeds_t=torch.stack([speeds_comp[key] for key in speeds_comp.keys()],dim=0)
success_rates_t = torch.stack([success_rates[key] for key in success_rates.keys()],dim=0)


create_polar_plot( speeds_t, [key for key in speeds_comp.keys()] ,'Speed (m/s)','speed_comparison')
create_polar_plot( success_rates_t,[key for key in success_rates.keys()] ,'Success Rate','success_rate_comparison.png')