import os
import re
import glob
import traceback

def modify_shell_scripts(num_step_list, old_string="fireflow", new_string="ralston"):
    # Find all shell files in the current directory
    shell_files = glob.glob("*.sh")
    
    if not shell_files:
        print("No shell files found in the current directory.")
        return
    
    print(f"Found {len(shell_files)} shell files to modify.")
    
    for file_path in shell_files:
        try:
            # Read the file content
            with open(file_path, 'r') as file:
                content = file.read()

            # Remove "CUDA_VISIBLE_DEVICES=0 " and "CUDA_VISIBLE_DEVICES=7 " from content
            modified_content = content.replace("CUDA_VISIBLE_DEVICES=0 ", "")
            modified_content = modified_content.replace("CUDA_VISIBLE_DEVICES=7 ", "")
            modified_content = modified_content.replace(old_string, new_string)
            
            new_commands = []
            for num_steps in num_step_list:
                modified_content = re.sub(r"--num_steps\s+\d+", f"--num_steps {num_steps}", modified_content)
                new_commands.append(modified_content)
            

            final_content = "\n\n".join(new_commands)
            
            # Write each modified content to a new file with num_steps in the filename
            for num_steps, command in zip(num_step_list, new_commands):
                new_file_path = f"{os.path.splitext(file_path)[0]}_num_steps_{num_steps}.sh"
                with open(new_file_path, 'w') as file:
                    file.write(command)
                print(f"Created: {new_file_path}")
            
            print(f"Modified: {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e.__class__.__name__}")
            traceback.print_exc()

if __name__ == "__main__":
    # Ask the user for a list of num_steps values
    old_string = input("Enter the string to be replaced (default: 'fireflow'): ") or "fireflow"
    new_string = input("Enter the new string (default: 'ralston'): ") or "ralston"
    num_steps_input = input("Enter a list of num_steps values (integers) separated by commas (e.g., 5, 8, 15, 25): ")
    num_steps_input = input("Enter a list of num_steps values separated by commas (e.g., 5,8,15,25): ")
    try:
        num_steps_list = [int(x.strip()) for x in num_steps_input.split(",")]
        modify_shell_scripts(num_steps_list, old_string, new_string)
    except ValueError:
        print("Invalid input. Please enter a list of integers separated by commas, e.g., 5,8,15,25.")