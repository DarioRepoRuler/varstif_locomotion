from graphs_gen import *


default_speed = load_tensor_from_csv('local_v' ,filename='results.csv')
power = load_tensor_from_csv('power',filename='results.csv')
energy = load_tensor_from_csv('energy',filename='results.csv')
cot = load_tensor_from_csv('cot',filename='results.csv')
create_polar_plot(default_speed,'Speed (m/s)','polar_speed_default.png')
create_polar_plot(power,'Power (W)','polar_power.png')
create_polar_plot(energy,'Energy (J)','polar_energy.png')
