import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const sourceRoot = path.resolve(process.env.BENCHMARK_SOURCE ?? path.join(here, '..', '..', '..'))
const output = path.resolve(here, '..', 'src', 'data')
const sourceFiles = {
  internals: 'benchmarks/results/benchmark_results.json',
  ollamaBaseline: 'benchmarks/results/ollama/baseline.json',
  contextScaling: 'benchmarks/results/ollama/context_scaling.json',
  outputScaling: 'benchmarks/results/ollama/output_scaling.json',
  modelComparison: 'benchmarks/results/ollama/model_comparison.json',
  concurrency: 'benchmarks/results/ollama/concurrency.json',
}

await mkdir(output, { recursive: true })
const records = Object.fromEntries(await Promise.all(Object.entries(sourceFiles).map(async ([key, file]) => {
  const contents = await readFile(path.join(sourceRoot, file), 'utf8')
  return [key, JSON.parse(contents)]
})))
const report = await readFile(path.join(sourceRoot, 'BENCHMARKS.md'), 'utf8')
const processor = report.match(/Processor\*\*:\s*(.+)/)?.[1]?.trim()
const gpu = report.match(/GPU Present\*\*:\s*(.+?)\s*\(([\d,]+)\s*MiB VRAM/)
if (!processor || !gpu) throw new Error('Could not read processor and GPU metadata from BENCHMARKS.md')
const psBlock = report.match(/NAME\s+ID\s+SIZE\s+PROCESSOR\s+CONTEXT\s+VRAM RESIDENCY\s*\n([\s\S]*?)\n```/)?.[1]
if (!psBlock) throw new Error('Could not read the ollama ps residency table from BENCHMARKS.md')
const gpuResidency = psBlock.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
  const match = line.match(/^(\S+)\s+\S+\s+[\d,.]+\s+\w+\s+(\d+%\s+GPU)\s+\d+\s+([\d,.]+)\s+MiB\s+\/\s+([\d,.]+)\s+MiB$/)
  if (!match) throw new Error(`Could not parse ollama ps row: ${line}`)
  return { model: match[1], processor: match[2], residentMiB: Number(match[3].replaceAll(',', '')), totalMiB: Number(match[4].replaceAll(',', '')) }
})
const hardware = { cpu: processor, gpu: gpu[1].trim(), vramMiB: Number(gpu[2].replaceAll(',', '')), runtime: 'Ollama 0.34.4 · CUDA offload', ollamaProcessor: gpuResidency[0].processor, gpuResidency }
const bundle = { generatedFrom: 'llm-inference-lab', sourceFiles, hardware, ...records }
await writeFile(path.join(output, 'benchmarks.generated.json'), `${JSON.stringify(bundle, null, 2)}\n`)
console.log(`Synced ${Object.keys(sourceFiles).length} benchmark artifacts from ${sourceRoot}`)
