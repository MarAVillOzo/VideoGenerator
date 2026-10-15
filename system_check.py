import platform
import subprocess

def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=30)
        return ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as e:
        return str(e)

print("=== LOCAL VIDEO STUDIO RTX 3060 CHECK ===")
print("Windows:", platform.platform())
print("Python:", platform.python_version())
print("\n--- NVIDIA ---")
print(run("nvidia-smi"))
print("\n--- CPU/RAM ---")
ps_cmd = "powershell -NoProfile -Command \"$cpu=Get-CimInstance Win32_Processor; $ram=(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum; Write-Host ('CPU: '+$cpu.Name); Write-Host ('Cores: '+$cpu.NumberOfCores); Write-Host ('RAM GB: '+[math]::Round($ram/1GB,2))\""
print(run(ps_cmd))
print("\n--- OLLAMA ---")
print(run("ollama --version"))
print(run("ollama list"))
print("\n--- DISK ---")
disk_cmd = "powershell -NoProfile -Command \"Get-Volume | Where-Object DriveLetter | Select DriveLetter,@{N='FreeGB';E={[math]::Round($_.SizeRemaining/1GB,1)}} | Format-Table -AutoSize\""
print(run(disk_cmd))
