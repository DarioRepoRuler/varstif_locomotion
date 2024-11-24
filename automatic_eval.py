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



# # Example call    
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


def modify_and_execute_script(config_path, checkpoint_path):
    """
    Modifies the config.yaml file to update the 'ckpt_path' property and executes test.py.

    Args:
        config_path (str): Path to the original config.yaml file.
        checkpoint_path (str): Path to the checkpoint to set in the config.
    """
    # Load the configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the checkpoint path
    config['ckpt_path'] = checkpoint_path

    # Add or modify additional configuration elements
    config['env']['sample_command_interval'] = 500
    config['env']['kick_vel'] = 0.0
    config['env']['terminate_geoms'] = ["base_0", "base_1", "base_2", "FR_hip", "FL_hip", "RR_hip", "RL_hip"]
    config['env']['enable_force_kick'] = False
    config['env']['impulse_force_kick'] = False
    config['env']['force_kick_impulse'] = [20.0, 20.0]
    config['env']['force_kick_interval']= 150
    config['env']['kick_force'] = [50.0,600.0]
    config['env']['is_training'] = False
    config['env']['domain_rand']['enable'] = False
    config['env']['control_range'] = {
        'cmd_x': [-1.5, 1.5],
        'cmd_y': [-1.5, 1.5],
        'cmd_ang': [-0.0, 0.0]
    }
    config['env']['manual_control'] = {
        'enable': True,
        'task': 'auto',
        'cmd_x': 0.8,
        'cmd_y': 0.0,
        'cmd_ang': 0.0
    }
    
    config['rollouts_per_experiment'] = 8
    config['success_threshold'] = 0.78125
    config['timesteps_per_rollout'] = 50
    config['plot_details']=False
    config['num_iterations'] = 65
    config['num_envs'] = 1000
    config['viz'] = False
    config['device']='cuda:0'
    print(f"Updated config with new properties and ckpt_path: {checkpoint_path}")

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

def process_dates_in_range(output_dir, start_date, end_date):
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
                    if 'checkpoints' in os.listdir(subfolder_path):
                        checkpoint_path = find_highest_checkpoint(os.path.join(subfolder_path, 'checkpoints'))
                        ## Old evluation script
                        if checkpoint_path:
                            config_path = find_hydra_config(subfolder_path)
                            if config_path:
                                modify_and_execute_script(config_path, checkpoint_path)
                            else:
                                print(f'No config.yaml found for checkpoint in {subfolder_path}')
                        elif checkpoint_path is None:
                            print(f'No checkpoints found in {subfolder_path}')
                    else:
                        print(f'No checkpoints in {subfolder_path}')
                        ## New evaluation script
                        # if checkpoint_path:
                        #         # Extract numerical part if the file matches 'model_{i}.pt'
                        #         filename = os.path.basename(checkpoint_path)
                        #         if filename.startswith("model_") and filename.endswith(".pt"):
                        #             try:
                        #                 checkpoint_num = int(filename.split('_')[1].split('.')[0])
                        #                 if checkpoint_num > 500:
                        #                     print(f"Found valid checkpoint {filename} with number > 500.")
                        #                     #modify_and_execute_script(config_path, checkpoint_path)
                        #                 else:
                        #                     print(f"Skipping ckpt {filename} as number is <= 500.")
                        #             except ValueError:
                        #                 print(f"Could not parse numerical part of {filename}, skipping.")
                        #         elif filename in ['last.pt', 'best.pt']:
                        #             print(f"Found {filename}, proceeding.")
                        #             #modify_and_execute_script(config_path, checkpoint_path)
                        # else:
                        #     print(f"Skipping checkpoint {filename} as it does not meet criteria.")

# Example usage
output_dir = os.path.join(os.getcwd(), 'outputs')
start_date = '2024-11-22'
end_date = '2024-11-22'

process_dates_in_range(output_dir, start_date, end_date)
