def get_scheduled_sampling_epsilon(
        current_step: int,
        decay_start_step: int = 10000,
        decay_end_step: int = 15000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.5
) -> float:
    """
    Calculates the value of epsilon for scheduled sampling using a linear decay schedule.

    This function allows you to control the probability of using the "teacher"
    (ground truth) versus the model's own predictions during training. The value
    of epsilon starts at epsilon_start, remains constant until decay_start_step,
    linearly decays to epsilon_end by decay_end_step, and then remains
    constant at epsilon_end.

    Args:
        current_step (int): The current training step or epoch.
        decay_start_step (int): The step at which the decay of epsilon begins.
        decay_end_step (int): The step at which the decay of epsilon finishes.
        epsilon_start (float, optional): The starting value of epsilon. Defaults to 1.0.
        epsilon_end (float, optional): The final value of epsilon after decay. Defaults to 0.0.

    Returns:
        float: The calculated epsilon value for the current step.

    Raises:
        ValueError: If decay_start_step is greater than decay_end_step.

    Example Usage:
        >>> # Epsilon will be 1.0 until step 10000
        >>> get_scheduled_sampling_epsilon(5000, 10000, 50000)
        1.0
        >>> # Epsilon will decay linearly between step 10000 and 50000
        >>> get_scheduled_sampling_epsilon(30000, 10000, 50000)
        0.5
        >>> # Epsilon will be 0.0 from step 50000 onwards
        >>> get_scheduled_sampling_epsilon(60000, 10000, 50000)
        0.0
    """
    if decay_start_step > decay_end_step:
        raise ValueError(
            f"decay_start_step ({decay_start_step}) cannot be greater than decay_end_step ({decay_end_step}).")

    # Before the decay period starts, return the starting epsilon
    if current_step < decay_start_step:
        return epsilon_start

    # After the decay period ends, return the final epsilon
    if current_step >= decay_end_step:
        return epsilon_end

    # During the decay period, perform linear interpolation
    total_decay_steps = decay_end_step - decay_start_step
    # Handle the edge case where start and end are the same to avoid division by zero
    if total_decay_steps == 0:
        return epsilon_end

    steps_into_decay = current_step - decay_start_step
    decay_fraction = steps_into_decay / total_decay_steps

    # Linearly interpolate from start to end
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * decay_fraction

    return epsilon


if __name__ == '__main__':
    # --- Example Demonstration ---
    # You can run this file directly to see a demonstration.

    DECAY_START = 10000
    DECAY_END = 50000
    TOTAL_STEPS = 60000

    print(f"--- Scheduled Sampling Epsilon Demonstration ---")
    print(f"Decay starts at step {DECAY_START} and ends at step {DECAY_END}.\n")

    test_steps = [0, 5000, DECAY_START, 20000, 30000, 40000, DECAY_END, 55000, TOTAL_STEPS]

    for step in test_steps:
        epsilon_val = get_scheduled_sampling_epsilon(step, DECAY_START, DECAY_END)
        print(f"Step {step:<6}: Epsilon = {epsilon_val:.4f}")

    # You can also uncomment the following lines to visualize the decay with matplotlib
    # try:
    #     import matplotlib.pyplot as plt
    #     steps = np.arange(0, TOTAL_STEPS)
    #     epsilons = [get_scheduled_sampling_epsilon(s, DECAY_START, DECAY_END) for s in steps]

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(steps, epsilons, label='Epsilon Value')
    #     plt.title('Scheduled Sampling Epsilon Decay')
    #     plt.xlabel('Training Step')
    #     plt.ylabel('Epsilon (Probability of using GT)')
    #     plt.axvline(x=DECAY_START, color='r', linestyle='--', label='Decay Start')
    #     plt.axvline(x=DECAY_END, color='g', linestyle='--', label='Decay End')
    #     plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    #     plt.legend()
    #     plt.show()
    # except ImportError:
    #     print("\nMatplotlib not found. Skipping visualization.")
    #     print("Install it with: pip install matplotlib")
