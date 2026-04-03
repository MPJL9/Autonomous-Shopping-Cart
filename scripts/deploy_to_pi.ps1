param(
  [string]$PiHost = "10.65.142.17",
  [string]$PiUser = "veda",
  [string]$RemoteDir = "~/Autonomous-Shopping-Cart-main"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archive = Join-Path $repoRoot "pi-deploy.tar.gz"
$sshTarget = "$PiUser@$PiHost"

if (Test-Path $archive) {
  Remove-Item -Force $archive
}

Write-Host "[deploy] Creating archive..."
tar --exclude=".git" --exclude=".pytest_cache" --exclude=".venv" --exclude="*.egg-info" --exclude=".env.pi" -czf $archive -C $repoRoot .

Write-Host "[deploy] Uploading archive to $sshTarget ..."
scp $archive "${sshTarget}:~/pi-deploy.tar.gz"

Write-Host "[deploy] Extracting on Pi..."
ssh $sshTarget "rm -rf $RemoteDir && mkdir -p $RemoteDir && tar -xzf ~/pi-deploy.tar.gz -C $RemoteDir"

Write-Host "[deploy] Running Pi setup..."
ssh $sshTarget "bash $RemoteDir/scripts/pi_setup.sh $RemoteDir"

Write-Host "[deploy] Done. Start the agent with:"
Write-Host "  ssh $sshTarget `"bash $RemoteDir/scripts/pi_run_agent.sh`""
