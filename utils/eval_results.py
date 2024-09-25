from graphs_gen import *


default_speed = load_tensor_from_csv('local_v' ,filename='results_pos.csv')
vic2_speed = load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv')
vic2_woar_speed = load_tensor_from_csv('local_v',filename='results_vic2_woar.csv')
vic2_default_speed = load_tensor_from_csv('local_v',filename='results_vic2.csv')
vic2_jt_speed = load_tensor_from_csv('local_v',filename='results_vic2_jt.csv')
vic2_jt_hard_speed = load_tensor_from_csv('local_v',filename='results_vic2_jt_hard.csv')
vic2_jz_middle_speed = load_tensor_from_csv('local_v', filename='results_vic2_jt_middle.csv')
print(default_speed)
print(vic2_speed)


success_rate_pos = load_tensor_from_csv('success_rate',filename='results_pos.csv')
success_rate_vic2 = load_tensor_from_csv('success_rate',filename='results_vic2.csv')
success_rate_vic2_woar = load_tensor_from_csv('success_rate',filename='results_vic2_woar.csv')



speeds=torch.stack([default_speed,vic2_speed, vic2_default_speed, vic2_woar_speed, vic2_jt_speed, vic2_jt_hard_speed,vic2_jz_middle_speed],dim=0)
success_rates = torch.stack([success_rate_pos,success_rate_vic2,success_rate_vic2_woar],dim=0)
print(speeds.shape)
create_polar_plot( speeds,['Baseline', 'VIC2 w ar feet air', 'VIC2 w ar default', 'VIC2 wo ar', 'VIC2 wo ar& w jt', 'VIC wo ar & w jt hard', 'VIC wo ar & w jt middle'] ,'Speed (m/s)','speed_comparison.png')
create_polar_plot(success_rates,['Baseline', 'VIC2 w ar', 'VIC2 wo ar'],'Success Rate','success_rate_comparison.png')