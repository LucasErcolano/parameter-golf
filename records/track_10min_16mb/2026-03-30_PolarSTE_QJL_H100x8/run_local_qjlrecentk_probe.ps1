param(
    [int]$MaxWallclockSeconds = 90,
    [int]$FinalizeBudgetSeconds = 10,
    [string]$RunId = "local_record_qjlrecentk8_probe",
    [string]$KvEvalCompareBackends = "qjl,qjl_recentk",
    [int]$KvRecentFp16Tokens = 8,
    [int]$KvQat = 0,
    [int]$KvQatStartStep = 64,
    [int]$KvQatRampSteps = 192,
    [int]$KvQatRecentTokens = 8,
    [double]$KvQatFarFraction = 1.0,
    [double]$KvQatMaxScale = 1.0,
    [int]$KvQatLayerStart = 0,
    [int]$KvQatLayerEnd = -1,
    [int]$KvEvalMaxTokens = 512,
    [int]$ValMaxTokens = 4096
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path $scriptDir -Parent))
$candidateRoots = @(
    $repoRoot,
    (Join-Path (Split-Path -Parent $repoRoot) "parameter-golf")
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
    throw "Could not locate fineweb10B_sp1024 dataset and tokenizer near $repoRoot"
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
$env:TRAIN_SEQ_LEN = "256"
$env:TRAIN_BATCH_TOKENS = "65536"
$env:ITERATIONS = "100000"
$env:WARMUP_STEPS = "0"
$env:LR_WARMUP_STEPS = "32"
$env:LR_WARMUP_INIT_SCALE = "0.1"
$env:TRAIN_LOG_EVERY = "50"
$env:VAL_LOSS_EVERY = "0"
$env:VAL_MAX_TOKENS = "$ValMaxTokens"
$env:EVAL_AUTOREGRESSIVE_KV = "1"
$env:KV_QUANT_BACKEND = "qjl"
$env:KV_EVAL_COMPARE_BACKENDS = "$KvEvalCompareBackends"
$env:KV_RECENT_FP16_TOKENS = "$KvRecentFp16Tokens"
$env:KV_QAT = "$KvQat"
$env:KV_QAT_START_STEP = "$KvQatStartStep"
$env:KV_QAT_RAMP_STEPS = "$KvQatRampSteps"
$env:KV_QAT_RECENT_TOKENS = "$KvQatRecentTokens"
$env:KV_QAT_FAR_FRACTION = "$KvQatFarFraction"
$env:KV_QAT_MAX_SCALE = "$KvQatMaxScale"
$env:KV_QAT_LAYER_START = "$KvQatLayerStart"
$env:KV_QAT_LAYER_END = "$KvQatLayerEnd"
$env:KV_EVAL_CONTEXT_LEN = "256"
$env:KV_EVAL_MAX_TOKENS = "$KvEvalMaxTokens"
$env:KV_BACKEND_SELFTEST = "0"
$env:ENABLE_TORCH_COMPILE = "0"
$env:MAX_WALLCLOCK_SECONDS = "$MaxWallclockSeconds"
$env:FINALIZE_BUDGET_SECONDS = "$FinalizeBudgetSeconds"
$env:LOG_SYNC_TO_DISK = "1"

Write-Host "Running $RunId with KV_QUANT_BACKEND=$env:KV_QUANT_BACKEND KV_EVAL_COMPARE_BACKENDS=$KvEvalCompareBackends KV_RECENT_FP16_TOKENS=$KvRecentFp16Tokens KV_QAT=$KvQat KV_QAT_RECENT_TOKENS=$KvQatRecentTokens KV_QAT_FAR_FRACTION=$KvQatFarFraction KV_QAT_MAX_SCALE=$KvQatMaxScale KV_QAT_LAYERS=$KvQatLayerStart`:$KvQatLayerEnd DATA_PATH=$dataRoot"
py -3.11 train_gpt.py
