# import os
# import subprocess
#
import os
import subprocess

"""
Batch evaluation script: Replaces the index in pred_file_name with the actual generated 
JSON file number range, and runs the evaluation command for each iteration.
"""

# Basic configuration
input_dir = r'/home/hdh/Desktop/zjj_Datas/Lite-AMNet/data/FreiHAND'
output_dir = r'/home/hdh/Desktop/zjj_Datas/Lite-AMNet/out'
base_pred_file = r'/home/hdh/Desktop/zjj_Datas/Lite-AMNet/out/MultipleDatasets/Lite_AMNet/Lite_AMNet{}.json'

eval_script_path = r'/home/hdh/Desktop/zjj_Datas/Lite-AMNet/tools/freihand-master/eval.py'

# Loop from 1 to 70 (code uses 0 to 3) to replace numbers and execute command
for i in range(0, 3):

    try:
        # Construct the prediction file path for the current round
        pred_file_name = base_pred_file.format(i)

        # Check if the file exists
        if not os.path.exists(pred_file_name):
            print(f"Warning: File {pred_file_name} does not exist, skipping this round")
            continue

        print(f"Executing evaluation for round {i}, using file: {pred_file_name}")

        # Construct the command to execute
        command = ['python', eval_script_path, input_dir, output_dir, '--pred_file_name', pred_file_name]

        # Execute the command and wait for completion
        subprocess.run(command, check=True)

        print(f"Evaluation for round {i} completed")

    except subprocess.CalledProcessError as e:
        print(f"Evaluation for round {i} failed: {e}")
    except Exception as e:
        print(f"Error occurred during evaluation for round {i}: {e}")

print("All rounds of evaluation completed")
