import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
RANDOM_ROOT = os.path.dirname(current_file_path)
sys.path.append(RANDOM_ROOT)

import eval_core

if __name__ == "__main__":
    eval_core.run_evaluation(
        test_name="map_size",
        param_name="size",
        param_values=[(10, 10), (20, 20), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80)],
        static_agents=10,
        static_size=None,
        static_density=0.1,
        num_episodes=20
    )