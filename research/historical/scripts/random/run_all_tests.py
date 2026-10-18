import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


SCRIPTS_TO_RUN = [
    "agent_count/evaluate_agents.py",
    "map_size/evaluate_map_size.py",
    "obstacle_density/evaluate_obstacle_density.py"
]

def main():
    print("==================================================")
    print("🚀 STARTING AUTOMATED MAPF TEST SUITE 🚀")
    print("==================================================")

    for script_rel_path in SCRIPTS_TO_RUN:
        script_full_path = os.path.join(BASE_DIR, script_rel_path)
        script_dir = os.path.dirname(script_full_path)
        script_name = os.path.basename(script_full_path)

        if not os.path.exists(script_full_path):
            print(f"\n[WARNING] File not found: {script_full_path}. Skipping.")
            continue

        print(f"\n{'-'*50}")
        print(f"▶ RUNNING: {script_rel_path}")
        print(f"{'-'*50}\n")

        try:
            
            
            subprocess.run(
                [sys.executable, script_name],
                cwd=script_dir,
                check=True 
            )
            print(f"\n✅ COMPLETED SUCCESSFULLY: {script_rel_path}")

        except subprocess.CalledProcessError as e:
            
            print(f"\n❌ [CRITICAL ERROR] Test {script_rel_path} failed (code: {e.returncode}).")
            print("Moving to the next test in the queue...")
        
        except KeyboardInterrupt:
            
            print("\n🛑 [INTERRUPTED] User interrupted the tests (Ctrl+C).")
            break

    print("\n==================================================")
    print("🏁 ALL QUEUED TESTS COMPLETED 🏁")
    print("==================================================")

if __name__ == "__main__":
    main()