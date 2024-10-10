import json
import os
import subprocess
import sys


def test_e2e_script():
    # Add the parent directory to the Python path (going up one level)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Simulate calling the script via command line using subprocess
    input_data = json.dumps([["2024-09-27", 45], ["2024-09-28", 55]])

    # Run the script as if from the command line, capturing stderr and stdout
    result = subprocess.run(
        [
            "python",
            "cyclical_encoding_new_inputs.py",
            input_data,
        ],
        capture_output=True,
        text=True,
        cwd=os.path.join(
            os.path.dirname(__file__), ".."
        ),  # Set cwd to the parent of 'tests' directory, which should be 'feature_engineering'
    )

    # Check if the process ran successfully
    assert result.returncode == 0, f"Subprocess failed with return code {result.returncode}"

    # Check the output (predicted cyclical features)
    output = result.stdout.strip()

    # Assert that output is not empty and matches expected format
    assert len(output) > 0, f"Output was empty. Error: {result.stderr}"
