param(
    [int]$NumKvHeads = 4,
    [int]$NumLayers = 9,
    [int]$NumLoops = 1,
    [int]$ModelDim = 512,
    [int]$NumHeads = 8,
    [int]$LoraRank = 0,
    [double]$LoraLr = 0.01,
    [double]$TiedEmbedLr = 0.05,
    [double]$MatrixLr = 0.04,
    [double]$ScalarLr = 0.04,
    [int]$KvTrainAux = 0,
    [string]$KvTrainAuxBackend = "qjl_ste",
    [double]$KvTrainAuxWeight = 0.25,
    [int]$KvTrainAuxTokens = 64,
    [int]$KvTrainAuxBatchSeqs = 4,
    [int]$KvTrainAuxEvery = 1,
    [int]$IterationEmbed = 0,
    [double]$IterationEmbedInitStd = 0.02,
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

if ($NumLoops -gt 1 -and -not $PSBoundParameters.ContainsKey("IterationEmbed")) {
    $IterationEmbed = 1
}

$modeLabel = if ($NumKvHeads -eq 1) { "mqa" } else { "gqa$NumKvHeads" }
if (-not $RunId) {
    $loopLabel = "l${NumLayers}x${NumLoops}"
    $adapterLabel = if ($LoraRank -gt 0) { "lora${LoraRank}" } else { "nolora" }
    $kvAuxLabel = if ($KvTrainAux -ne 0) { "kvaux" } else { "nokvaux" }
    $iterLabel = if ($IterationEmbed -ne 0) { "iter" } else { "plain" }
    $RunId = "local_qjl_gap_${modeLabel}_${loopLabel}_d${ModelDim}_${adapterLabel}_${kvAuxLabel}_${iterLabel}_${MaxWallclockSeconds}s"
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
$env:NUM_LAYERS = "$NumLayers"
$env:NUM_LOOPS = "$NumLoops"
$env:MODEL_DIM = "$ModelDim"
$env:NUM_HEADS = "$NumHeads"
$env:NUM_KV_HEADS = "$NumKvHeads"
$env:LORA_RANK = "$LoraRank"
$env:LORA_LR = "$LoraLr"
$env:TIED_EMBED_LR = "$TiedEmbedLr"
$env:MATRIX_LR = "$MatrixLr"
$env:SCALAR_LR = "$ScalarLr"
$env:KV_TRAIN_AUX = "$KvTrainAux"
$env:KV_TRAIN_AUX_BACKEND = "$KvTrainAuxBackend"
$env:KV_TRAIN_AUX_WEIGHT = "$KvTrainAuxWeight"
$env:KV_TRAIN_AUX_TOKENS = "$KvTrainAuxTokens"
$env:KV_TRAIN_AUX_BATCH_SEQS = "$KvTrainAuxBatchSeqs"
$env:KV_TRAIN_AUX_EVERY = "$KvTrainAuxEvery"
$env:ITERATION_EMBED = "$IterationEmbed"
$env:ITERATION_EMBED_INIT_STD = "$IterationEmbedInitStd"
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

Write-Host "Running $RunId with NUM_LAYERS=$NumLayers NUM_LOOPS=$NumLoops MODEL_DIM=$ModelDim NUM_HEADS=$NumHeads NUM_KV_HEADS=$NumKvHeads LORA_RANK=$LoraRank LORA_LR=$LoraLr TIED_EMBED_LR=$TiedEmbedLr MATRIX_LR=$MatrixLr SCALAR_LR=$ScalarLr KV_TRAIN_AUX=$KvTrainAux KV_TRAIN_AUX_BACKEND=$KvTrainAuxBackend KV_TRAIN_AUX_WEIGHT=$KvTrainAuxWeight KV_TRAIN_AUX_TOKENS=$KvTrainAuxTokens KV_TRAIN_AUX_BATCH_SEQS=$KvTrainAuxBatchSeqs KV_TRAIN_AUX_EVERY=$KvTrainAuxEvery ITERATION_EMBED=$IterationEmbed DATA_PATH=$dataRoot"
py -3.11 train_gpt.py
