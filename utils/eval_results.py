from graphs_gen import *


default_speed = load_tensor_from_csv('local_v' ,filename='results_pos.csv')
vic2_speed = load_tensor_from_csv('local_v',filename='results_vic2_feet_air.csv')
vic2_woar_speed = load_tensor_from_csv('local_v',filename='results_vic2_woar.csv')
vic2_default_speed = load_tensor_from_csv('local_v',filename='results_vic2.csv')
print(default_speed)
print(vic2_speed)
# power = load_tensor_from_csv('power',filename='results.csv')
# energy = load_tensor_from_csv('energy',filename='results.csv')
# cot = load_tensor_from_csv('cot',filename='results.csv')
# create_polar_plot(default_speed,'Speed (m/s)','polar_speed_default.png')
# create_polar_plot(power,'Power (W)','polar_power.png')
# create_polar_plot(energy,'Energy (J)','polar_energy.png')

success_rate_pos = load_tensor_from_csv('success_rate',filename='results_pos.csv')
success_rate_vic2 = load_tensor_from_csv('success_rate',filename='results_vic2.csv')
success_rate_vic2_woar = load_tensor_from_csv('success_rate',filename='results_vic2_woar.csv')


speeds=torch.stack([default_speed,vic2_speed, vic2_default_speed, vic2_woar_speed],dim=0)
success_rates = torch.stack([success_rate_pos,success_rate_vic2,success_rate_vic2_woar],dim=0)
print(speeds.shape)
create_polar_plot( speeds,['Baseline', 'VIC2 w ar feet air', 'VIC2 w ar default', 'VIC2 wo ar'] ,'Speed (m/s)','speed_comparison.png')
create_polar_plot(success_rates,['Baseline', 'VIC2 w ar', 'VIC2 wo ar'],'Success Rate','success_rate_comparison.png')