# Launch a command outside the Claude app's process tree (parent: WmiPrvSE), hidden.
param([string]$Command, [string]$Directory)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $Command; CurrentDirectory = $Directory; ProcessStartupInformation = $si}
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
