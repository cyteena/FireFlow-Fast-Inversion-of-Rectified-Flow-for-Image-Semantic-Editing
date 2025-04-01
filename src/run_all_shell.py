import os
import glob
import subprocess

def run_all_shell_scripts():
    # Find all shell files in the current directory
    shell_files = glob.glob("*_num_steps_*.sh")
    
    if not shell_files:
        print("No shell files found in the current directory.")
        return
    
    print(f"Found {len(shell_files)} shell files to execute.")
    
    for file_path in shell_files:
        try:
            print(f"Running: {file_path}")
            # Run the shell script
            result = subprocess.run(["bash", file_path], check=True, text=True, capture_output=True)
            print(f"Output of {file_path}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error while running {file_path}:\n{e.stderr}")

if __name__ == "__main__":
    run_all_shell_scripts()