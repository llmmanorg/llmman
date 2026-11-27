# Builds llmman-desktop's Windows package with MakeAppx (Windows SDK):
#
#   packaging/desktop-msix.ps1 pack   <x64|arm64> <version> <dir> <out.msix>
#   packaging/desktop-msix.ps1 bundle <version> <dir> <out.msixbundle>
#   packaging/desktop-msix.ps1 verify <version> <bundle>
#
# pack:   one .msix from <dir>, which holds llmman-desktop.exe and the
#         llmman.exe for the same architecture.
# bundle: the .msixbundle of every .msix in <dir>.
# verify: checks a bundle has both architectures, each with both programs.
#
# <version> is MAJOR.MINOR.PATCH; ".0" is added for MSIX.
# $env:MSIX_PUBLISHER is the manifest Publisher (default CN=llmman); signed,
#   it must be the certificate's subject.
# $env:MSIX_UNSIGNED=true marks the package unsigned: `Add-AppxPackage
#   -AllowUnsigned` refuses one without the OID this adds to the Publisher.
# $env:MAKEAPPX is the makeappx.exe to use, else the newest in the SDK.

param(
    [Parameter(Mandatory)][string]$Command,
    [Parameter(ValueFromRemainingArguments)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Find-MakeAppx {
    if ($env:MAKEAPPX) { return $env:MAKEAPPX }
    $bin = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $exe = Get-ChildItem $bin -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Join-Path $_.FullName 'x64\makeappx.exe' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $exe) { throw "makeappx.exe not found under $bin" }
    $exe
}

function Invoke-MakeAppx {
    & (Find-MakeAppx) @args
    if ($LASTEXITCODE -ne 0) { throw "makeappx $($args[0]) failed ($LASTEXITCODE)" }
}

function Assert-Version([string]$Version) {
    if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "version `"$Version`" is not MAJOR.MINOR.PATCH" }
}

function Get-Publisher {
    $publisher = if ($env:MSIX_PUBLISHER) { $env:MSIX_PUBLISHER } else { 'CN=llmman' }
    if ($env:MSIX_UNSIGNED -eq 'true') { $publisher += ', OID.2.25.311729368913984317654407730594956997722=1' }
    $publisher
}

# Pinned downloads from the docs-assets release; desktop/generate-assets.py
# makes them and prints the sha256s. Staged without the llmman-desktop- prefix.
$LogoPins = [ordered]@{
    'llmman-desktop-Square44x44Logo.png'   = '7419a3a32ff1b400e149dffd35c86c86044a6ec6c0c58d51a38458510a416cd5'
    'llmman-desktop-Square150x150Logo.png' = '4af49818d8b50b42b89d06b81b472c6a51ae85e22d926d6e46bae07ecd14b6ea'
    'llmman-desktop-StoreLogo.png'         = '5ab5f2af20f0e482523db8053634bbd248e5f9b87f7f97c7bdc44605dc7a4f09'
}

function Get-Logos([string]$Dir) {
    foreach ($name in $LogoPins.Keys) {
        $path = Join-Path $Dir ($name -replace '^llmman-desktop-', '')
        for ($try = 1; ; $try++) {
            try {
                Invoke-WebRequest "https://github.com/llmmanorg/llmman/releases/download/docs-assets/$name" -OutFile $path -UseBasicParsing
                break
            }
            catch {
                if ($try -ge 4) { throw "could not download ${name}: $_" }
                Start-Sleep -Seconds (2 * $try)
            }
        }
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
        if ($hash -ne $LogoPins[$name]) { throw "$name has sha256 $hash, not the $($LogoPins[$name]) pinned in this script" }
    }
}

function New-TempDir {
    $dir = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid())
    New-Item -ItemType Directory -Path $dir | Out-Null
    $dir
}

function Invoke-Pack {
    param(
        [Parameter(Mandatory)][ValidateSet('x64', 'arm64')][string]$Arch,
        [Parameter(Mandatory)][string]$Version,
        [Parameter(Mandatory)][string]$Dir,
        [Parameter(Mandatory)][string]$Out
    )
    Assert-Version $Version
    foreach ($exe in 'llmman-desktop.exe', 'llmman.exe') {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $exe))) { throw "$Dir has no $exe" }
    }
    $stage = New-TempDir
    try {
        New-Item -ItemType Directory -Path (Join-Path $stage 'Assets') | Out-Null
        Copy-Item (Join-Path $Dir 'llmman-desktop.exe'), (Join-Path $Dir 'llmman.exe') $stage
        Get-Logos (Join-Path $stage 'Assets')
        $manifest = (Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'msix\AppxManifest.xml.in')).
            Replace('@VERSION@', "$Version.0").
            Replace('@ARCH@', $Arch).
            Replace('@PUBLISHER@', [Security.SecurityElement]::Escape((Get-Publisher)))
        [IO.File]::WriteAllText((Join-Path $stage 'AppxManifest.xml'), $manifest, [Text.UTF8Encoding]::new($false))
        Invoke-MakeAppx pack /d $stage /p $Out /o
    }
    finally {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}

function Invoke-Bundle {
    param(
        [Parameter(Mandatory)][string]$Version,
        [Parameter(Mandatory)][string]$Dir,
        [Parameter(Mandatory)][string]$Out
    )
    Assert-Version $Version
    Invoke-MakeAppx bundle /d $Dir /p $Out /bv "$Version.0" /o
}

function Invoke-Verify {
    param(
        [Parameter(Mandatory)][string]$Version,
        [Parameter(Mandatory)][string]$Bundle
    )
    Assert-Version $Version
    $tmp = New-TempDir
    try {
        Invoke-MakeAppx unbundle /p $Bundle /d (Join-Path $tmp 'bundle') /o
        $seen = @()
        foreach ($package in Get-ChildItem (Join-Path $tmp 'bundle') -Recurse -File -Filter *.msix) {
            $dest = Join-Path $tmp $package.BaseName
            Invoke-MakeAppx unpack /p $package.FullName /d $dest /o
            foreach ($file in 'AppxManifest.xml', 'llmman-desktop.exe', 'llmman.exe') {
                if (-not (Test-Path -LiteralPath (Join-Path $dest $file))) { throw "$($package.Name) has no $file" }
            }
            $identity = ([xml](Get-Content -Raw -LiteralPath (Join-Path $dest 'AppxManifest.xml'))).Package.Identity
            if ($identity.Version -ne "$Version.0") { throw "$($package.Name) is version $($identity.Version), not $Version.0" }
            if ($identity.Publisher -ne (Get-Publisher)) { throw "$($package.Name) has Publisher $($identity.Publisher), not $(Get-Publisher)" }
            $seen += $identity.ProcessorArchitecture
        }
        if ((($seen | Sort-Object) -join ',') -ne 'arm64,x64') {
            throw "the bundle holds [$($seen -join ', ')], expected one x64 and one arm64 package"
        }
        Write-Host "OK: $Bundle holds x64 and arm64 packages of $Version, each with llmman-desktop.exe and llmman.exe"
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force
    }
}

switch ($Command) {
    'pack' { Invoke-Pack @Rest }
    'bundle' { Invoke-Bundle @Rest }
    'verify' { Invoke-Verify @Rest }
    default { throw "usage: desktop-msix.ps1 pack|bundle|verify ... (see the header)" }
}
