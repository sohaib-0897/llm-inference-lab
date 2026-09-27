export interface BenchmarkRecord {
  scenario: string
  batch_size: number
  prompt_len: number
  gen_len: number
  use_cache: boolean
  ttft_ms: number
  avg_tpot_ms: number
  p95_tpot_ms: number
  total_latency_ms: number
  tokens_per_second: number
  aggregate_throughput: number
  kv_cache_mb: number
  process_memory_mb: number
  extra: Record<string, unknown>
}

export interface OllamaBaseline {
  experiment: string
  model: string
  trials_count: number
  mean_ttft_ms: number
  mean_total_latency_ms: number
  p95_total_latency_ms: number
  mean_client_tokens_per_sec: number
  mean_server_tokens_per_sec: number
}

export interface ModelComparison {
  model_name: string
  param_size: string
  quantization: string
  disk_size_mb: number
  vram_size_mb: number
  mean_ttft_ms: number
  mean_prompt_eval_ms: number
  mean_eval_tokens_per_sec: number
  mean_client_tokens_per_sec: number
  mean_total_latency_ms: number
}

export interface ContextScaling {
  target_approx_tokens: number
  actual_prompt_tokens: number
  output_tokens: number
  mean_prompt_eval_duration_ms: number
  mean_client_ttft_ms: number
  mean_total_latency_ms: number
  mean_tokens_per_sec: number
}

export interface OutputScaling {
  requested_output_tokens: number
  actual_output_tokens: number
  mean_client_ttft_ms: number
  mean_eval_duration_ms: number
  mean_total_latency_ms: number
  mean_tokens_per_sec: number
}

export interface ConcurrencyResult {
  concurrency: number
  aggregate_tokens_per_sec: number
  wall_time_ms: number
  total_tokens_generated: number
  p95_latency_ms: number
  mean_latency_ms: number
  median_latency_ms: number
  mean_ttft_ms: number
  per_request_mean_throughput: number
}

export interface BenchmarkBundle {
  generatedFrom: string
  sourceFiles: Record<string, string>
  hardware: HardwareInfo
  internals: BenchmarkRecord[]
  ollamaBaseline: OllamaBaseline[]
  contextScaling: ContextScaling[]
  outputScaling: OutputScaling[]
  modelComparison: ModelComparison[]
  concurrency: ConcurrencyResult[]
}

export interface HardwareInfo {
  cpu: string
  gpu: string
  vramMiB: number
  runtime: string
  ollamaProcessor: string
  gpuResidency: GPUResidencyObservation[]
}

export interface GPUResidencyObservation {
  model: string
  processor: string
  residentMiB: number
  totalMiB: number
}

export interface KVCacheResult {
  promptTokens: number
  generatedTokens: number
  cached: BenchmarkRecord
  recompute: BenchmarkRecord
}

export interface BatchingResult {
  batchSize: number
  aggregateTokensPerSecond: number
  totalLatencyMs: number
}

export interface QuantizationResult {
  precision: string
  footprintMB: number
  tokensPerSecond: number
  latencyMs: number
}

export interface OnnxComparison {
  engine: string
  forwardLatencyMs: number
  speedup: number
}
