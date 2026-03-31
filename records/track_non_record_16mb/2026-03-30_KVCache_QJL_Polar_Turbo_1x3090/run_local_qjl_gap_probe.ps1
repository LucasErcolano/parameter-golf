param(
    [int]$NumKvHeads = 1,
    [int]$MaxWallclockSeconds = 90,
    [int]$FinalizeBudgetSeconds = 10,
    [string]$RunId = "",
    [int]$KvEvalMaxTokens = 512,
    [int]$ValMaxTokens = 4096,
    [int]$WarmdownIters = 0
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))
$candidateRoots = @(
    (Join-Path (Split-Path -Parent $workspaceRoot) "parameter-golf"),
    (Join-Path (Split-Path -Parent $workspaceRoot) "parameter-golf-main"),
    $workspaceRoot
)

$dataRoot = $null
$tokenizerPath = $null
foreach ($root in $candidateRoots) {
    $candidateData = Join-Path $root "data\\datasets\\fineweb10B_sp1024"
    $candidateTokenizer = Join-Path $root "data\\tokenizers\\fineweb_1024_bpe.model"
    if ((Test-Path -LiteralPath $candidateData) -and (Test-Path -LiteralPath $candidateTokenizer)) {
        $dataRoot = $candidateData
        $tokenizerPath = $candidateTokenizer
        break
    }
}

if (-not $dataRoot -or -not $tokenizerPath) {
    throw "Could not locate fineweb10B_sp1024 dataset and tokenizer near $workspaceRoot"
}

$modeLabel = if ($NumKvHeads -eq 1) { "mqa" } else { "gqa$NumKvHeads" }
if (-not $RunId) {
    $RunId = "local_qjl_gap_${modeLabel}_${MaxWallclockSeconds}s"
}

$env:RUN_ID = $RunId
$env:DATA_PATH = $dataRoot
$env:TOKENIZER_PATH = $tokenizerPath
$env:VOCAB_SIZE = "1024"
$env:QAT = "1"
$env:QAT_SCHEME = "polar"
$env:WEIGHT_QUANT_SCHEME = "polar"
$env:POLAR_QAT_BITS_MODE = "quality"
$env:POLAR_WEIGHT_BITS_MODE = "quality"
$env:POLAR_WEIGHT_ROTATE = "0"
$env:NUM_KV_HEADS = "$NumKvHeads"
$env:TRAIN_SEQ_LEN = "256"
$env:TRAIN_BATCH_TOKENS = "65536"
$env:ITERATIONS = "100000"
$env:WARMDOWN_ITERS = "$WarmdownIters"
$env:WARMUP_STEPS = "0"
$env:LR_WARMUP_STEPS = "32"
$env:LR_WARMUP_INIT_SCALE = "0.1"
$env:TRAIN_LOG_EVERY = "50"
$env:VAL_LOSS_EVERY = "0"
$env:VAL_MAX_TOKENS = "$ValMaxTokens"
$env:EVAL_AUTOREGRESSIVE_KV = "1"
$env:KV_QUANT_BACKEND = "qjl"
$env:KV_EVAL_CONTEXT_LEN = "256"
$env:KV_EVAL_MAX_TOKENS = "$KvEvalMaxTokens"
$env:KV_BACKEND_SELFTEST = "0"
$env:ENABLE_TORCH_COMPILE = "0"
$env:MAX_WALLCLOCK_SECONDS = "$MaxWallclockSeconds"
$env:FINALIZE_BUDGET_SECONDS = "$FinalizeBudgetSeconds"
$env:LOG_SYNC_TO_DISK = "1"

Write-Host "Running $RunId with NUM_KV_HEADS=$NumKvHeads DATA_PATH=$dataRoot"
py -3.11 train_gpt.py
