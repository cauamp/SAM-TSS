import subprocess
import os

# SSH connection details
ssh_host = "narval3"
remote_path = "/lustre06/project/6085198/caua/sam-tss/run_*"
local_path = os.path.expanduser("./runs/")  # Local directory to save the downloaded files

# Create local directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

# rsync command to download, excluding .pth files
cmd = [
    "rsync",
    "-avzh",
    "--exclude=*.pth*",
    f"{ssh_host}:{remote_path}",
    local_path
]

try:
    subprocess.run(cmd, check=True)
    print(f"Download completed to {local_path}")
except subprocess.CalledProcessError as e:
    print(f"Error during download: {e}")