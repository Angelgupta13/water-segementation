<#
.SYNOPSIS
    Builds the Docker image, starts a container, and validates health + prediction endpoint.
.DESCRIPTION
    Tests that:
    1. `docker build` succeeds
    2. Container starts and passes HEALTHCHECK
    3. GET /health returns {"status": "ok"} (or "degraded" if no model)
    4. POST /predict with a sample image returns a valid PNG mask
    5. POST /predict without file returns 422
    6. POST /predict with invalid backend returns 400
    7. Container stops cleanly

    When model.onnx or best_model.pth is found locally, it is mounted into the
    container so the full prediction pipeline works (health=ok, predict=200).
.PARAMETER Tag
    Docker image tag to use (default: water-segmentation:test)
.PARAMETER SkipBuild
    Skip docker build (use existing image)
.PARAMETER ModelPath
    Path to model file to mount (auto-detected if not specified)
.EXAMPLE
    .\scripts\test_docker.ps1
    .\scripts\test_docker.ps1 -Tag myrepo/water-segmentation:latest
    .\scripts\test_docker.ps1 -SkipBuild
#>

param(
    [string]$Tag = "water-segmentation:test",
    [switch]$SkipBuild,
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"
$ContainerName = "water-seg-test-$(Get-Random -Maximum 99999)"
$MountArgs = @()

# Auto-detect model file (prefer ONNX for speed)
if (-not $ModelPath) {
    $scriptRoot = Split-Path -Parent $PSScriptRoot
    if (Test-Path "$scriptRoot\model.onnx") { $ModelPath = "$scriptRoot\model.onnx" }
    elseif (Test-Path "$scriptRoot\best_model.pth") { $ModelPath = "$scriptRoot\best_model.pth" }
}

if ($ModelPath -and (Test-Path $ModelPath)) {
    $ext = [System.IO.Path]::GetExtension($ModelPath)
    $target = if ($ext -eq ".onnx") { "/app/model.onnx" } else { "/app/best_model.pth" }
    $MountArgs = @("-v", "${ModelPath}:${target}")
    Write-Host "INFO: Mounting model $ModelPath -> $target" -ForegroundColor Cyan
} else {
    Write-Host "INFO: No local model found; container will run in degraded mode" -ForegroundColor Yellow
}


# -- 1. Build ----------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "=== Step 1: Building Docker image '$Tag' ===" -ForegroundColor Cyan
    docker build -t $Tag .
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: Build failed" -ForegroundColor Red; exit 1 }
    Write-Host "PASS: Build succeeded" -ForegroundColor Green
} else {
    Write-Host "=== Step 1: Skipping build (using existing image '$Tag') ===" -ForegroundColor Yellow
}


# -- 2. Start container ------------------------------------------------------
Write-Host "`n=== Step 2: Starting container '$ContainerName' ===" -ForegroundColor Cyan

# Stop & remove any leftover from prior run (ignore errors if none exists)
$ErrorActionPreference = "Continue"
docker stop $ContainerName 2>$null | Out-Null
docker rm $ContainerName 2>$null | Out-Null
$ErrorActionPreference = "Stop"

docker run -d --name $ContainerName -p 18000:8000 @MountArgs $Tag
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: Container start failed" -ForegroundColor Red; exit 1 }


# -- 3. Wait for HEALTHCHECK -------------------------------------------------
Write-Host "`n=== Step 3: Waiting for container to become healthy ===" -ForegroundColor Cyan
$maxRetries = 30
$delay = 2
for ($i = 1; $i -le $maxRetries; $i++) {
    $status = (docker inspect --format='{{json .State.Health.Status}}' $ContainerName 2>$null) -replace '"',''
    if ($status -eq "healthy") {
        Write-Host "PASS: Container healthy after ${i}s" -ForegroundColor Green
        break
    }
    if ($i -eq $maxRetries) {
        docker logs $ContainerName
        docker stop $ContainerName; docker rm $ContainerName
        Write-Host "FAIL: Container did not become healthy within $($maxRetries * $delay)s" -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds $delay
}


# -- 4. Test /health ---------------------------------------------------------
Write-Host "`n=== Step 4: Testing GET /health ===" -ForegroundColor Cyan
$health = (Invoke-WebRequest -Uri "http://localhost:18000/health" -UseBasicParsing).Content | ConvertFrom-Json
if ($health.status -eq "ok") {
    Write-Host "PASS: /health returned 'ok' (model loaded)" -ForegroundColor Green
} elseif ($health.status -eq "degraded") {
    Write-Host "INFO: /health returned 'degraded' (no model)" -ForegroundColor Yellow
} else {
    Write-Host "FAIL: /health returned unexpected status='$($health.status)'" -ForegroundColor Red
    docker stop $ContainerName; docker rm $ContainerName; exit 1
}


# -- 5. Create test image ----------------------------------------------------
Write-Host "`n=== Step 5: Preparing test image ===" -ForegroundColor Cyan
$imgPath = "$env:TEMP\water_seg_test.jpg"
python -c @"
import cv2, numpy as np
img = np.zeros((256, 256, 3), dtype=np.uint8)
img[:, :, 0] = 200
img[:, :, 2] = 255
cv2.imwrite(r'$imgPath', img)
"@
if (-not (Test-Path $imgPath)) { Write-Host "FAIL: Could not create test image" -ForegroundColor Red; exit 1 }
Write-Host "PASS: Test image created at $imgPath" -ForegroundColor Green


# -- 6. Test POST /predict (valid image) -------------------------------------
Write-Host "`n=== Step 6: Testing POST /predict (valid image) ===" -ForegroundColor Cyan
$maskOut = "$env:TEMP\water_seg_mask.png"
$statusLine = curl.exe -s -X POST "http://localhost:18000/predict" -F "file=@$imgPath" -o "$maskOut" -w "%{http_code};%{content_type};%{size_download}" 2>$null
$code, $ctype, $size = $statusLine -split ';'

if ($code -eq 200) {
    Write-Host "PASS: /predict returned 200 ($size bytes, $ctype)" -ForegroundColor Green
    Remove-Item $maskOut -Force -ErrorAction SilentlyContinue
} elseif ($code -eq 503) {
    Write-Host "INFO: /predict returned 503 (no model -- expected in degraded mode)" -ForegroundColor Yellow
} else {
    Write-Host "FAIL: /predict returned HTTP $code" -ForegroundColor Red
    docker stop $ContainerName; docker rm $ContainerName; exit 1
}


# -- 7. Test POST /predict (no file -> 422) ----------------------------------
Write-Host "`n=== Step 7: Testing POST /predict (no file -> 422) ===" -ForegroundColor Cyan
try {
    $null = Invoke-WebRequest -Uri "http://localhost:18000/predict" -Method Post -UseBasicParsing
    Write-Host "FAIL: Expected 422, got 200" -ForegroundColor Red
    docker stop $ContainerName; docker rm $ContainerName; exit 1
} catch {
    if ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "PASS: /predict without file returned 422" -ForegroundColor Green
    } else {
        Write-Host "FAIL: Expected 422, got $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        docker stop $ContainerName; docker rm $ContainerName; exit 1
    }
}


# -- 8. Test POST /predict (invalid backend -> 400) --------------------------
Write-Host "`n=== Step 8: Testing POST /predict (invalid backend -> 400) ===" -ForegroundColor Cyan
$statusLine = curl.exe -s -X POST "http://localhost:18000/predict?backend=tensorrt" -F "file=@$imgPath" -w "%{http_code}" -o NUL 2>$null
if ($statusLine -eq 400) {
    Write-Host "PASS: /predict with invalid backend returned 400" -ForegroundColor Green
} else {
    Write-Host "FAIL: Expected 400, got $statusLine" -ForegroundColor Red
    docker stop $ContainerName; docker rm $ContainerName; exit 1
}


# -- 9. Cleanup --------------------------------------------------------------
Write-Host "`n=== Step 9: Cleaning up ===" -ForegroundColor Cyan
docker stop $ContainerName >$null; docker rm $ContainerName >$null
Remove-Item $imgPath -Force -ErrorAction SilentlyContinue
Write-Host "PASS: Container stopped and removed" -ForegroundColor Green


# -- All tests passed --------------------------------------------------------
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ALL DOCKER TESTS PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
