import os 
import shutil
from datetime import datetime
import subprocess
import yaml

def check_for_checkpoints_and_delete(day_outputs):
    for file in os.listdir(day_outputs):
        subfolder_path = os.path.join(day_outputs, file)
        if os.path.isdir(subfolder_path):
            if 'checkpoints' not in os.listdir(subfolder_path):
                print(f"Deleting {subfolder_path} as it doesn't contain 'checkpoints'")
                #shutil.rmtree(subfolder_path)

def delete_all_non_checkpoints_runs(output_dir):
    for day in os.listdir(output_dir):
        day_outputs = os.path.join(output_dir,day)
        delete_non_checkpoint_run(day_outputs)
        
def delete_non_checkpoint_run(day_outputs):
    if os.path.exists(day_outputs):
        print(f'{day_outputs} exists')
        check_for_checkpoints_and_delete(day_outputs)
    else:
        assert False, f'{day_outputs} does not exist'

def find_highest_checkpoint(checkpoints_path):
    """
    Finds the checkpoint file with the highest numerical suffix (e.g., model_10.pt).

    Args:
        checkpoints_path (str): Path to the checkpoints directory.

    Returns:
        str: Path to the checkpoint file with the highest number (or None if none found).
    """
    highest_checkpoint = None
    highest_number = -1
    for filename in os.listdir(checkpoints_path):
        if filename == 'best.pt':
            print(f'Found best.pt in {os.path.join(checkpoints_path, filename)}')
            return os.path.join(checkpoints_path, filename)
        elif filename == 'last.pt':
            print(f'Found last.pt in {os.path.join(checkpoints_path, filename)}')
            return os.path.join(checkpoints_path, filename)
        elif filename.startswith('model_') and filename.endswith('.pt'):
            try:
                number = int(filename.split('_')[1].split('.')[0])
                if number > highest_number:
                    highest_number = number
                    highest_checkpoint = os.path.join(checkpoints_path, filename)
            except ValueError:
                pass  # Ignore non-numeric filenames
    if highest_checkpoint is not None:
        print(f'Found highest checkpoint {highest_checkpoint}')
    return highest_checkpoint

def find_hydra_config(subfolder_path):
    """
    Finds the config.yaml file in the .hydra folder if it exists.

    Args:
        subfolder_path (str): Path to the subfolder containing .hydra.

    Returns:
        str: Path to the config.yaml file (or None if not found).
    """
    hydra_path = os.path.join(subfolder_path, '.hydra', 'config.yaml')
    if os.path.exists(hydra_path):
        print(f'Found config.yaml in {hydra_path}')
        return hydra_path
    else:
        print(f'No config.yaml found in {os.path.join(subfolder_path, ".hydra")}')
        return None


def find_highest_checkpoint_in_date_range(output_dir, start_date, end_date):
    """
    Finds the highest checkpoint within a specified date range.

    Args:
        output_dir (str): Path to the output directory.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        str: Path to the highest checkpoint file (or None if none found).
    """
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    highest_checkpoint = None

    for day in os.listdir(output_dir):
        day_dir = os.path.join(output_dir, day)
        try:
            day_date_obj = datetime.strptime(day, '%Y-%m-%d')
        except ValueError:
            # Ignore non-date formatted directories
            continue

        if start_date_obj <= day_date_obj <= end_date_obj:
            for file in os.listdir(day_dir):
                subfolder_path = os.path.join(day_dir, file)
                if os.path.isdir(subfolder_path) and 'checkpoints' in os.listdir(subfolder_path):
                    checkpoint_path = find_highest_checkpoint(os.path.join(subfolder_path, 'checkpoints'))
                    if checkpoint_path:
                        # Compare checkpoints based on your criteria (e.g., modification time, filename)
                        # Here, we'll assume the latest checkpoint is the best:
                        if not highest_checkpoint or os.path.getmtime(checkpoint_path) > os.path.getmtime(highest_checkpoint):
                            highest_checkpoint = checkpoint_path

    return highest_checkpoint


# Function to recursively update the configuration
def recursive_update(d, u):
    """
    Recursively update dictionary d with values from dictionary u.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def modify_and_execute_test_script(config_path, checkpoint_path, config_properties=None):
    """
    Modifies the config.yaml file to update the 'ckpt_path' property and executes test.py.

    Args:
        config_path (str): Path to the original config.yaml file.
        checkpoint_path (str): Path to the checkpoint to set in the config.
        config_properties (dict): Optional dictionary of additional config properties to modify or add.
    """
    # Load the configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the checkpoint path
    config['ckpt_path'] = checkpoint_path

    # # Add or modify additional configuration elements if provided
    # if config_properties:
    #     config.update(config_properties)
    # Recursively update the configuration with the provided properties
    if config_properties:
        config = recursive_update(config, config_properties)


    # Save the modified configuration to a temporary file
    temp_config_dir = os.path.join(os.getcwd(), 'temp_config')
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_config_dir, 'test.yaml')
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Execute test.py with the modified configuration
    try:
        command = (
            f"XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python test.py "
            f"--config-path {temp_config_dir} --config-name test"
        )
        print(f"Executing command: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
    finally:
        # Clean up temporary configuration directory
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if os.path.exists(temp_config_dir) and not os.listdir(temp_config_dir):
            os.rmdir(temp_config_dir)

def eval_trained_model(subfolder_path, config_properties = None):

    if 'checkpoints' in os.listdir(subfolder_path):
        checkpoint_path = find_highest_checkpoint(os.path.join(subfolder_path, 'checkpoints'))
        ## Old evluation script
        if checkpoint_path:
            config_path = find_hydra_config(subfolder_path)
            # Extract numerical part if the file matches 'model_{i}.pt'
            filename = os.path.basename(checkpoint_path)
            if filename.startswith("model_") and filename.endswith(".pt"):
                try:
                    checkpoint_num = int(filename.split('_')[1].split('.')[0])
                    if checkpoint_num > 500:
                        print(f"Found valid checkpoint {filename} with number > 500.")
                        modify_and_execute_test_script(config_path, checkpoint_path, config_properties = config_properties)
                    else:
                        print(f"Skipping ckpt {filename} as number is <= 500.")
                except ValueError:
                    print(f"Could not parse numerical part of {filename}, skipping.")
            elif filename in ['last.pt', 'best.pt']:
                print(f"Found {filename}, proceeding.")
                modify_and_execute_test_script(config_path, checkpoint_path, config_properties = config_properties)
        elif checkpoint_path is None:
            print(f'No highest nor best/last checkpoint found in {subfolder_path}')
    else:
        print(f'No checkpoints in {subfolder_path}')



def process_dates_in_range(output_dir, start_date, end_date, config_changes = None):
    """
    Processes subdirectories within a date range, checking for and deleting non-checkpoint runs.

    Args:
        output_dir (str): Path to the output directory.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
    """
    
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    for day in os.listdir(output_dir):
        day_dir = os.path.join(output_dir, day)
        try:
            day_date_obj = datetime.strptime(day, '%Y-%m-%d')
        except ValueError:
            # Ignore non-date formatted directories
            continue

        if start_date_obj <= day_date_obj <= end_date_obj:
            print(f"Processing directory for date: {day}")
            for file in os.listdir(day_dir):
                subfolder_path = os.path.join(day_dir, file)
                if os.path.isdir(subfolder_path):
                    
                    eval_trained_model(subfolder_path, config_changes)


## Example usage

# ----------Evaluation of model in a date range ----------- #
output_dir = os.path.join(os.getcwd(), 'outputs')
start_date = '2024-12-06'
end_date = '2024-12-06'
config_changes = {
        'env': {
            'sample_command_interval': 500,
            'kick_vel': 0.0,
            'terminate_geoms': ["base_0", "base_1", "base_2", "FR_hip", "FL_hip", "RR_hip", "RL_hip"],
            'enable_force_kick': False,
            'impulse_force_kick': False,
            'force_kick_impulse': [20.0, 20.0],
            'force_kick_interval': 150,
            'kick_force': [50.0, 300.0],
            'is_training': False,
            'domain_rand': {
                'enable': False
            },
            'control_range': {
                'cmd_x': [-1.5, 1.5],
                'cmd_y': [-1.5, 1.5],
                'cmd_ang': [-0.0, 0.0]
            },
            'manual_control': {
                'enable': True,
                'task': 'auto',
                'cmd_x': 0.8,
                'cmd_y': 0.0,
                'cmd_ang': 0.0
            }
        },
        'rollouts_per_experiment': 8,
        'success_threshold': 0.78125,
        'timesteps_per_rollout': 50,
        'plot_details': False,
        'num_iterations': 65,
        'num_envs': 1000,
        'viz': False,
        'record_video':False,
        'result_tag': "test_automation",
        'device': 'cuda:0'
    }
#process_dates_in_range(output_dir, start_date, end_date, config_changes)



# ---------------- Example call ---------------- #    
# output_dir = os.path.join(os.getcwd(),'outputs')
# for day in os.listdir(output_dir):
#     day_outputs = os.path.join(output_dir,day)
#     for file in os.listdir(day_outputs):
#         subfolder_path = os.path.join(day_outputs, file)
#         if os.path.isdir(subfolder_path):
#             if 'checkpoints' in os.listdir(subfolder_path):
#                 checkpoint_path = find_highest_checkpoint(os.path.join(subfolder_path, 'checkpoints'))
#                 if checkpoint_path is None:
#                     print(f'No checkpoints found in {subfolder_path}')
#             else:
#                 print(f'No checkpoints in {subfolder_path}')

# # day = '2024-11-23'
# #delete_non_checkpoint_run(os.path.join(output_dir,day))
# #delete_all_non_checkpoints_runs(output_dir)

# # ----------Evaluation of single model ----------- #
# trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-24/09-35-09'
# config_changes = {
#         'env': {
#             'sample_command_interval': 500,
#             'kick_vel': 0.0,
#             'terminate_geoms': ["base_0", "base_1", "base_2", "FR_hip", "FL_hip", "RR_hip", "RL_hip"],
#             'enable_force_kick': False,
#             'impulse_force_kick': False,
#             'force_kick_impulse': [20.0, 20.0],
#             'force_kick_interval': 150,
#             'kick_force': [50.0, 300.0],
#             'is_training': False,
#             'domain_rand': {
#                 'enable': False
#             },
#             'control_range': {
#                 'cmd_x': [-1.5, 1.5],
#                 'cmd_y': [-1.5, 1.5],
#                 'cmd_ang': [-0.0, 0.0]
#             },
#             'manual_control': {
#                 'enable': True,
#                 'task': 'stiffness',
#                 'cmd_x': 0.8,
#                 'cmd_y': 0.0,
#                 'cmd_ang': 0.0
#             }
#         },
#         'rollouts_per_experiment': 8,
#         'success_threshold': 0.78125,
#         'timesteps_per_rollout': 50,
#         'plot_details': False,
#         'num_iterations': 65,
#         'num_envs': 1000,
#         'viz': True,
#         'record_video':True,
#         'result_tag': "test_automation",
#         'device': 'cuda:0'
#     }

# #eval_trained_model(trained_run_path, config_changes)
# config_changes['scene_xml'] = 'unitree_go2/flat.xml'
# config_changes['result_tag'] = 'test'
# config_changes['env']['manual_control']['cmd_x'] = 0.5
# eval_trained_model(trained_run_path, config_changes)
# trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-24/09-58-49'
# eval_trained_model(trained_run_path, config_changes)
# trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-25/14-03-45'
# eval_trained_model(trained_run_path, config_changes)



config_changes = {
        'env': {
            'sample_command_interval': 500,
            'kick_vel': 0.0,
            'terminate_geoms': [],
            'enable_force_kick': True,
            'impulse_force_kick': False,
            'force_kick_duration': 0.2,
            'force_kick_impulse': [20.0, 20.0],
            'force_kick_interval': 150,
            'kick_force': [50.0, 300.0],
            'kick_theta': [0.0, 2.0], # kick_theta * pi
            'is_training': False,
            'domain_rand': {
                'randomisation': False
            },
            'control_range': {
                'cmd_x': [-1.5, 1.5],
                'cmd_y': [-1.5, 1.5],
                'cmd_ang': [-0.0, 0.0]
            },
            'manual_control': {
                'enable': True,
                'task': 'force_push',
                'cmd_x': 0.0,
                'cmd_y': 0.0,
                'cmd_ang': 0.0
            }
        },
        'rollouts_per_experiment': 5,
        'success_threshold': 0.78125,
        'timesteps_per_rollout': 50,
        'plot_details': False,
        'num_iterations': 21,
        'num_envs': 1000,
        'viz': False,
        'record_video':False,
        'result_tag': "test_stand",
        'device': 'cuda:0'
    }
config_changes['scene_xml'] = 'unitree_go2/flat.xml'


#trained_run_path = '/home/dspoljaric/TAvic/outputs/2024-11-25/10-20-58'
#eval_trained_model(trained_run_path, config_changes)
trained_run_path = '/home/dspoljaric/TAvic/outputs/2024-11-25/14-03-45'
eval_trained_model(trained_run_path, config_changes)
trained_run_path = '/home/dspoljaric/TAvic/outputs/2024-11-24/09-58-49'
eval_trained_model(trained_run_path, config_changes)

# for i in range(7):
#     config_changes['env']['manual_control']['cmd_x'] = 1.0 + i*0.1
#     config_changes['result_tag'] = f'test_vel_1_{i}'
#     #eval_trained_model(trained_run_path, config_changes)
#     trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-25/14-03-45'
#     eval_trained_model(trained_run_path, config_changes)
#     trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-24/09-58-49'
#     eval_trained_model(trained_run_path, config_changes)

# trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-25/14-03-45'
# eval_trained_model(trained_run_path, config_changes)

# trained_run_path = '/home/dario/Documents/TALocoMotion/outputs/2024-11-24/09-58-49'
# eval_trained_model(trained_run_path, config_changes)
