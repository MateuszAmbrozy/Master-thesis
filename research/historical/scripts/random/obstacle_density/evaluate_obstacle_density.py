import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
RANDOM_ROOT = os.path.dirname(current_file_path)
sys.path.append(RANDOM_ROOT)

import eval_core

if __name__ == "__main__":
    eval_core.run_evaluation(
        test_name="obstacle_density",
        param_name="density",
        param_values=[0.0, 0.1, 0.2, 0.3],
        static_agents=8,
        static_size=(20, 20),
        static_density=None,
        num_episodes=20
    )