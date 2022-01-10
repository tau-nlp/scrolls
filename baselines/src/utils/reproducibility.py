import os
import subprocess
import zipfile
import logging

logger = logging.getLogger(__name__)


def save_git_info(output_dir, repo_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if repo_dir is None:
        repo_dir = os.getcwd()

    # Get to the top level git dir
    process = subprocess.run("git rev-parse --show-toplevel".split(), cwd=repo_dir, capture_output=True)
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    repo_dir = p_out

    git_instructions_file_path = os.path.join(output_dir, "git_instructions.txt")
    git_diff_file_path = os.path.join(output_dir, "git_diff.patch")
    git_untracked_file_path = os.path.join(output_dir, "untracked_files.zip")

    process = subprocess.run("git rev-parse --verify HEAD".split(), cwd=repo_dir, capture_output=True)
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    commit_hash = p_out

    process = subprocess.run(
        "git ls-files --other --full-name --exclude-standard".split(),
        cwd=repo_dir,
        capture_output=True,
    )
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    with zipfile.ZipFile(git_untracked_file_path, "w", zipfile.ZIP_DEFLATED) as f:
        for file_path in [x.strip() for x in p_out.split("\n") if len(x.strip()) > 0]:
            actual_file_path = os.path.join(repo_dir, file_path)
            if os.path.getsize(actual_file_path) > 1024 ** 2:
                logger.info(f"Git saving: Untracked file {file_path} is over 1MB, skipping")
            else:
                f.write(actual_file_path, file_path)

    with open(git_instructions_file_path, mode="w") as f:
        f.writelines(
            [
                "To restore the code, use the following commands in the repository (-b <new_branch_name> is optional):\n",
                f"git checkout -b <new_branch_name> {commit_hash}\n",
                f"git apply {git_diff_file_path}\n",
                f"unzip {git_untracked_file_path}",
            ]
        )
    with open(git_diff_file_path, mode="w") as f:
        subprocess.run("git diff HEAD --binary".split(), cwd=repo_dir, stdout=f, check=True)
