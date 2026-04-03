from __future__ import annotations

import argparse
import io
import posixpath
import sys
import tarfile
from pathlib import Path

import paramiko


EXCLUDES = {".git", ".pytest_cache", ".venv", "__pycache__", "autonomous_shopping_cart.egg-info", ".env.pi"}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDES for part in path.parts)


def build_archive(repo_root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in repo_root.rglob("*"):
            if should_skip(path):
                continue
            arcname = path.relative_to(repo_root)
            archive.add(path, arcname=str(arcname))
    buffer.seek(0)
    return buffer.read()


def run_command(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    status = stdout.channel.recv_exit_status()
    return status, stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="veda")
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-dir", default="~/Autonomous-Shopping-Cart-main")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    archive_bytes = build_archive(repo_root)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, look_for_keys=False, allow_agent=False, timeout=15)

    try:
        sftp = client.open_sftp()
        remote_archive = f"/home/{args.user}/pi-deploy.tar.gz"
        with sftp.file(remote_archive, "wb") as remote_file:
            remote_file.write(archive_bytes)
        sftp.close()

        commands = [
            f"rm -rf {args.remote_dir} && mkdir -p {args.remote_dir}",
            f"tar -xzf {remote_archive} -C {args.remote_dir}",
            f"PI_SUDO_PASS='{args.password}' bash {posixpath.join(args.remote_dir, 'scripts/pi_setup.sh')} {args.remote_dir}",
            "hostname -I | awk '{print $1}'",
        ]

        for command in commands[:-1]:
            status, out, err = run_command(client, command)
            print(f"$ {command}")
            if out.strip():
                print(out.strip())
            if err.strip():
                print(err.strip())
            if status != 0:
                raise RuntimeError(f"Remote command failed with exit code {status}: {command}")

        status, out, err = run_command(client, commands[-1])
        if status == 0 and out.strip():
            print(f"Pi IP: {out.strip()}")
        if err.strip():
            print(err.strip())
    finally:
        client.close()


if __name__ == "__main__":
    main()
