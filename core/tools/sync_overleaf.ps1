param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("push", "pull", "status")]
    [string]$Action = "push",

    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "Update paper files from workspace"
)

$envFile = Join-Path $PSScriptRoot "..\..\segredos.env"
if (-not (Test-Path $envFile)) {
    Write-Error "Arquivo segredos.env nao encontrado na raiz do workspace!"
    exit 1
}

$envVars = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        $envVars[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$projectId = $envVars["OVERLEAF_PROJECT_ID"]
$token = $envVars["OVERLEAF_TOKEN"]

if (-not $projectId -or -not $token) {
    Write-Error "OVERLEAF_PROJECT_ID ou OVERLEAF_TOKEN nao definidos em segredos.env!"
    exit 1
}

$authUrl = "https://git:$($token)@git.overleaf.com/$projectId"
$papersDir = Join-Path $PSScriptRoot "..\..\papers"
$tempCloneDir = Join-Path $env:TEMP "overleaf_sync_$projectId"

Write-Host "=== Sincronizacao Overleaf ($Action) ===" -ForegroundColor Cyan

if ($Action -eq "push") {
    Write-Host "[1/3] Clonando versao mais recente do Overleaf..."
    if (Test-Path $tempCloneDir) { Remove-Item -Path $tempCloneDir -Recurse -Force }
    git clone $authUrl $tempCloneDir --quiet

    Write-Host "[2/3] Copiando arquivos locais de papers/ para o Overleaf..."
    Get-ChildItem -Path $papersDir -Exclude ".git", "CONTEXT.md" | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $tempCloneDir -Recurse -Force
    }

    Write-Host "[3/3] Enviando alteracoes para o Overleaf via Git..."
    Push-Location $tempCloneDir
    git config user.name "Caua Lira"
    git config user.email "ccunhalira8760@gmail.com"
    git add -A
    $status = git status --porcelain
    if ($status) {
        git commit -m $CommitMessage
        git push origin main
        Write-Host "Sucesso! Arquivos atualizados no Overleaf com sucesso." -ForegroundColor Green
    } else {
        Write-Host "Nenhuma alteracao detectada. Overleaf ja esta sincronizado." -ForegroundColor Yellow
    }
    Pop-Location
    Remove-Item -Path $tempCloneDir -Recurse -Force -ErrorAction SilentlyContinue
}
elseif ($Action -eq "pull") {
    Write-Host "[1/2] Baixando alteracoes do Overleaf..."
    if (Test-Path $tempCloneDir) { Remove-Item -Path $tempCloneDir -Recurse -Force }
    git clone $authUrl $tempCloneDir --quiet

    Write-Host "[2/2] Atualizando pasta papers/ local..."
    Get-ChildItem -Path $tempCloneDir -Exclude ".git" | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $papersDir -Recurse -Force
    }
    Remove-Item -Path $tempCloneDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Sucesso! Pasta papers/ atualizada com o conteudo do Overleaf." -ForegroundColor Green
}
