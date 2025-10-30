# SPDX-License-Identifier: Apache-2.0

"""Script to check copyright and rights reserved headers in Python files.

This script provides methods to verify copyright and rights reserved headers
in Python files. It includes methods to check individual files as well as
multiple files within specified directories.

Methods:
    - check_file_headers: Verifies copyright and rights reserved headers in a
      specified Python file.
    - check_copyright: Checks copyright and rights reserved headers in Python
      files within specified directories.

Usage:
    This script can be executed from the command line, accepting optional paths
    to specific files or directories to check. If no paths are provided, it
    checks the current directory by default.

Examples:
    To check Python files in the current directory:
        $ poetry run python check_copyright.py

    To check Python files in a specific directory:
        $ poetry run python check_copyright.py path/to/directory

    To check specific Python files:
        $ poetry run python check_copyright.py path/to/file.py path/to/another/file.py

    The script will print error messages for any files with missing or incorrect
    copyright and rights reserved headers, and exits with a status code of 1 if
    any issues are found.
"""

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_file_headers(file_path: str) -> Optional[str]:
    """Verify copyright and rights reserved headers in a Python file.

    This method reads the specified Python file line by line and checks if each line
    matches the expected copyright and rights reserved headers. If any header is
    incorrect or missing, it returns an error message; otherwise, it returns None.

    Args:
        file_path: The path to the Python file.

    Returns:
        An error message if headers are incorrect or missing, or None
        if headers are correct.
    """
    expected_headers = ["# SPDX-License-Identifier: Apache-2.0"]
    with open(file_path) as file:
        for index, line in enumerate(file):
            line = line.strip()
            if index >= len(expected_headers):
                break
            if line != expected_headers[index]:
                return (
                    f"Error: {file_path} does not have the expected "
                    f"``{expected_headers[index]}`` line {index} header."
                )

        if index < len(expected_headers) - 1:
            return f"Error: {file_path} is missing a header."

    return None


def check_copyright(paths: List[str]) -> None:
    """Check copyright and rights reserved headers in Python files.

    This method scans Python files within specified directories and checks if they
    contain the correct copyright and rights reserved headers. If any header is missing
    or incorrect, it prints an error message and exits with a status code of 1.

    Args:
        paths: List of paths (files and directories) to check.

    Returns:
        None
    """
    file_paths = []
    for path in paths:
        if os.path.isfile(path):
            file_paths.append(path)
        elif os.path.isdir(path):
            dir_file_paths = glob.glob(os.path.join(path, "**/*.py"), recursive=True)
            file_paths.extend(dir_file_paths)

    file_paths = [file_path for file_path in file_paths if file_path.endswith(".py")]
    file_paths = [file_path for file_path in file_paths if "__init__" not in file_path]

    for file_path in file_paths:
        error_message = check_file_headers(file_path)
        if error_message:
            logger.error(error_message)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Specify the paths (files and directories) to check. Default is the "
        "current directory.",
    )
    args = parser.parse_args()

    check_copyright(args.paths)
