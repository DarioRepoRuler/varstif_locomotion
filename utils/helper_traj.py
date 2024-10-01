import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def create_sinusoidal_command(amplitude, frequency, episode_length, sampling_rate):
    """
    Creates a sinusoidal command with the given amplitude, frequency, episode length, and sampling rate.

    Args:
        amplitude: The maximum value of the sinusoid.
        frequency: The frequency of the sinusoid in Hz.
        episode_length: The duration of the episode in seconds.
        sampling_rate: The sampling rate in Hz.

    Returns:
        A JAX array containing the sinusoidal command.
    """

    num_timesteps = int(episode_length * sampling_rate)
    time_values = jnp.linspace(0, episode_length, num_timesteps)
    sinusoidal_values = amplitude * jnp.sin(2 * jnp.pi * frequency * time_values)

    return jnp.stack([time_values, sinusoidal_values], axis=1)

def create_combined_command(amplitude, frequency, episode_length, sampling_rate, cmd_x, cmd_y):
    """
    Creates a combined command with a sinusoidal command and constant cmd_x and cmd_y values.

    Args:
        amplitude: The maximum value of the sinusoid.
        frequency: The frequency of the sinusoid in Hz.
        episode_length: The duration of the episode in seconds.
        sampling_rate: The sampling rate in Hz.
        cmd_x: The constant cmd_x value.
        cmd_y: The constant cmd_y value.

    Returns:
        A JAX array containing the combined command.
    """

    sinusoidal_command = create_sinusoidal_command(amplitude, frequency, episode_length, sampling_rate)
    #print(f"sinusoidal command shape: {sinusoidal_command.shape}")
    combined_command = jnp.hstack([jnp.expand_dims(sinusoidal_command[:,0], 1),   jnp.full((sinusoidal_command.shape[0], 1), cmd_x),jnp.full((sinusoidal_command.shape[0], 1), cmd_y), jnp.expand_dims(sinusoidal_command[:,1], 1)])

    return combined_command


def main():
    # Set parameters
    amplitude = 0.4  # Amplitude of the sinusoid
    frequency = 1.0  # Frequency of the sinusoid
    episode_length = 10.0  # Episode length
    sampling_rate = 50  # Sampling rate (20 Hz)

    # Create the combined command
    cmd_x = 0.8  # Keep cmd_x constant
    cmd_y = 0.0  # Keep cmd_y constant
    combined_command = create_combined_command(amplitude, frequency, episode_length, sampling_rate, cmd_x, cmd_y)
    print(f"combined command: {combined_command}")
    # Extract time, cmd_x, and ang_v values
    time_values = combined_command[:, 0]
    cmd_x_values = combined_command[:, 2]
    cmd_y_values = combined_command[:, 3]
    ang_v_values = combined_command[:, 1]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, ang_v_values, label='cmd_x')
    plt.plot(time_values, cmd_x_values, label='cmd_y')
    plt.plot(time_values, cmd_y_values, label='ang_v')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Sinusoidal Command')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()