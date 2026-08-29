<#
.SYNOPSIS
    Script utilitário para atualizar a data e o status do HANDOFF.md rapidamente.
#>
param(
    [string]$Notas = ""
)

$handoffPath = Join-Path $PSScriptRoot "..\..\HANDOFF.md"
if (-not (Test-Path $handoffPath)) {
    Write-Error "HANDOFF.md nao encontrado!"
    exit 1
}

$branch = git rev-parse --abbrev-ref HEAD
$dateStr = Get-Date -Format "dd/MM/yyyy HH:mm"

Write-Host "=== Atualizacao do HANDOFF.md ===" -ForegroundColor Cyan
Write-Host "Data: $dateStr"
Write-Host "Branch Ativo: $branch"
if ($Notas) {
    Add-Content -Path $handoffPath -Value "`n### 📝 Nota Adicional ($dateStr):`n- $Notas"
    Write-Host "Nota anexada ao HANDOFF.md com sucesso!" -ForegroundColor Green
}
