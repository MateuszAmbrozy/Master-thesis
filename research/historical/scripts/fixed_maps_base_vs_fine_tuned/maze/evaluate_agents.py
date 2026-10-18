import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
RANDOM_ROOT = os.path.dirname(current_file_path)
sys.path.append(RANDOM_ROOT)

import eval_core

if __name__ == "__main__":        
    if len(sys.argv) < 2:
        print("ERROR: A testing mode is required!")
        print("Usage: python evaluate_agents.py <mode: classic | improved>")
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    custom_map_path = "maze-32-32-4.map"
    
    MODELS_BASE = str(eval_core.PROJECT_ROOT / "checkpoints")
   
    if mode == "classic":
        generated_test_name = "ft_vs_base_CLASSIC_maze"
        use_current_arch = False
        CHECKPOINT_BASE = os.path.join(MODELS_BASE, "base", "classic.pt") 
        CHECKPOINT_FT = os.path.join(MODELS_BASE, "fine_tuned", "classic", "maze.pt")
    elif mode == "improved":
        generated_test_name = "ft_vs_base_IMPROVED_maze"
        use_current_arch = True
        CHECKPOINT_BASE = os.path.join(MODELS_BASE, "base", "improved.pt") 
        CHECKPOINT_FT = os.path.join(MODELS_BASE, "fine_tuned", "improved", "maze.pt")
    else:
        print("ERROR: Allowed modes are 'classic' or 'improved' only")
        sys.exit(1)

    eval_core.run_evaluation(
        test_name=generated_test_name,
        param_name="agents",
        param_values=[4, 8, 16, 32],
        custom_map_path=custom_map_path,
        model_ft_path=CHECKPOINT_FT, 
        model_base_path=CHECKPOINT_BASE, 
        use_current_architecture=use_current_arch,
        num_episodes=20
    )