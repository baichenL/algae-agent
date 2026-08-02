import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid2 as Grid,
  Link,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import DownloadRounded from '@mui/icons-material/DownloadRounded'
import ExpandMoreRounded from '@mui/icons-material/ExpandMoreRounded'
import PrintRounded from '@mui/icons-material/PrintRounded'
import VerifiedRounded from '@mui/icons-material/VerifiedRounded'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { api, ApiError } from './api'
import type {
  EvaluationMetric,
  EvaluationReport,
  EvaluationSuite,
} from './types'
import { PageHeader } from './components'

echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

function formatMetric(metric?: EvaluationMetric): string {
  if (!metric || metric.status !== 'completed' || metric.value == null) return 'NOT RUN'
  if (metric.unit === 'ratio') return `${(metric.value * 100).toFixed(1)}%`
  if (metric.unit === 'milliseconds') return `${metric.value.toFixed(1)} ms`
  if (metric.unit === 'tokens') return Math.round(metric.value).toLocaleString()
  return Number(metric.value).toPrecision(4)
}

function suiteByKey(report: EvaluationReport, key: string): EvaluationSuite | undefined {
  return report.suites.find(suite => suite.key === key)
}

function suiteMetric(report: EvaluationReport, suiteKey: string, metricKey: string): EvaluationMetric | undefined {
  return suiteByKey(report, suiteKey)?.metrics.find(metric => metric.key === metricKey)
}

function statusColor(status: string): 'success' | 'warning' | 'error' | 'default' {
  if (status === 'completed' || status === 'passed') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'not_run') return 'warning'
  return 'default'
}

function MetricTile({ metric, label }: { metric?: EvaluationMetric; label: string }) {
  return <Card sx={{ height: '100%' }}>
    <CardContent>
      <Stack direction="row" justifyContent="space-between" gap={1} alignItems="flex-start">
        <Typography color="text.secondary" variant="body2">{label}</Typography>
        <Chip
          size="small"
          label={metric?.evidence_mode === 'llm_judge' ? 'LLM Judge' : 'Deterministic'}
          color={metric?.evidence_mode === 'llm_judge' ? 'warning' : 'success'}
          variant="outlined"
        />
      </Stack>
      <Typography variant="h4" fontWeight={800} mt={1}>{formatMetric(metric)}</Typography>
      <Typography variant="caption" color="text.secondary">
        n={metric?.denominator ?? '—'}
        {metric?.ci95 ? ` · 95% CI ${metric.unit === 'ratio' ? `${(metric.ci95[0] * 100).toFixed(1)}–${(metric.ci95[1] * 100).toFixed(1)}%` : `${metric.ci95[0].toPrecision(3)}–${metric.ci95[1].toPrecision(3)}`}` : ''}
      </Typography>
      {metric?.status_reason && <Typography color="warning.main" variant="caption" display="block" mt={1}>{metric.status_reason}</Typography>}
    </CardContent>
  </Card>
}

function RagAblationChart({ report }: { report: EvaluationReport }) {
  const comparison = report.comparisons.find(item => item.key === 'rag_retrieval_ablation')
  const variants = comparison?.variants || {}
  const names = ['fts', 'vector', 'hybrid', 'hybrid_reranker'].filter(name => variants[name])
  if (!names.length) return <Alert severity="warning">RAG Gold 尚未完成人工审核，消融结果为 NOT RUN。</Alert>
  const option = {
    tooltip: { trigger: 'axis' },
    legend: {},
    grid: { left: 55, right: 20, top: 45, bottom: 55 },
    xAxis: { type: 'category', data: names },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    series: [
      { name: 'Recall@20', type: 'bar', data: names.map(name => variants[name]?.recall_at_20) },
      { name: 'MRR@10', type: 'bar', data: names.map(name => variants[name]?.mrr_at_10) },
      { name: 'nDCG@10', type: 'bar', data: names.map(name => variants[name]?.ndcg_at_10) },
    ],
  }
  return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 350 }} />
}

function ComparisonChart({ report }: { report: EvaluationReport }) {
  const comparisons = report.comparisons.filter(item => ['scientific_replan_ablation', 'scientific_optimizer_ablation', 'context_compression_ablation'].includes(item.key))
  return <Grid container spacing={1.5}>{comparisons.map(comparison => {
    const variants: any[] = comparison.variants || []
    const isContext = comparison.key === 'context_compression_ablation'
    const option = {
      tooltip: { trigger: 'axis' },
      legend: isContext ? {} : undefined,
      grid: { left: 55, right: isContext ? 55 : 20, top: isContext ? 45 : 20, bottom: 55 },
      xAxis: { type: 'category', data: variants.map(variant => variant.name), axisLabel: { interval: 0, rotate: 15 } },
      yAxis: isContext
        ? [
            { type: 'value', name: 'tokens' },
            { type: 'value', name: 'ms' },
          ]
        : { type: 'value' },
      series: isContext
        ? [
            { name: 'Input tokens', type: 'bar', data: variants.map(variant => variant.tokens_mean), itemStyle: { color: '#3b82f6' } },
            { name: 'P95 latency', type: 'line', yAxisIndex: 1, data: variants.map(variant => variant.p95_latency_ms), itemStyle: { color: '#c87821' } },
          ]
        : [{
            type: 'bar',
            data: variants.map(variant => variant.value),
            itemStyle: { color: comparison.key.includes('optimizer') ? '#c87821' : '#0f766e' },
          }],
    }
    return <Grid key={comparison.key} size={{ xs: 12, lg: 4 }}>
      <Typography fontWeight={700} variant="body2">{comparison.display_name}</Typography>
      <ReactEChartsCore echarts={echarts} option={option} style={{ height: 270 }} />
    </Grid>
  })}</Grid>
}

function ConfusionMatrix({ report }: { report: EvaluationReport }) {
  const router = suiteByKey(report, 'agent')?.details?.router
  const labels: string[] = router?.labels || []
  const matrix: number[][] = router?.confusion_matrix || []
  if (!labels.length) return null
  const data = matrix.flatMap((row, y) => row.map((value, x) => [x, y, value]))
  const maximum = Math.max(1, ...data.map(item => Number(item[2])))
  const option = {
    tooltip: { formatter: (item: any) => `实际 ${labels[item.value[1]]}<br/>预测 ${labels[item.value[0]]}: ${item.value[2]}` },
    grid: { left: 120, right: 35, top: 25, bottom: 100 },
    xAxis: { type: 'category', data: labels, axisLabel: { rotate: 40 } },
    yAxis: { type: 'category', data: labels },
    visualMap: { min: 0, max: maximum, calculable: true, orient: 'horizontal', left: 'center', bottom: 0 },
    series: [{ type: 'heatmap', data, label: { show: true } }],
  }
  return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 470 }} />
}

function FailureSlices({ report }: { report: EvaluationReport }) {
  const slices: any[] = suiteByKey(report, 'agent')?.details?.failure_slices || []
  if (!slices.some(item => Number(item.failed) > 0)) {
    return <Alert severity="success">当前 deterministic Agent Gold 无失败；失败切片保留为零，未用默认通过替代缺失样本。</Alert>
  }
  const option = {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 25, bottom: 100 },
    xAxis: { type: 'category', data: slices.map(item => `${item.dimension}:${item.slice}`), axisLabel: { rotate: 40 } },
    yAxis: { type: 'value', minInterval: 1 },
    series: [{ name: '失败数', type: 'bar', data: slices.map(item => item.failed), itemStyle: { color: '#dc6039' } }],
  }
  return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 360 }} />
}

function MetricDetails({ suite }: { suite: EvaluationSuite }) {
  return <Accordion disableGutters>
    <AccordionSummary expandIcon={<ExpandMoreRounded />}>
      <Stack direction="row" spacing={1.5} alignItems="center" width="100%">
        <Typography fontWeight={750}>{suite.display_name}</Typography>
        <Chip size="small" label={suite.status.toUpperCase()} color={statusColor(suite.status)} />
        <Typography variant="caption" color="text.secondary">n={suite.sample_count}</Typography>
      </Stack>
    </AccordionSummary>
    <AccordionDetails>
      {suite.description && <Alert severity="info" sx={{ mb: 2 }}>{suite.description}</Alert>}
      <Stack spacing={1.5}>
        {suite.metrics.map(metric => <Paper key={metric.key} variant="outlined" sx={{ p: 2 }}>
          <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={1}>
            <Box>
              <Typography fontWeight={750}>{metric.display_name} · {formatMetric(metric)}</Typography>
              <Typography variant="body2" color="text.secondary">{metric.definition}</Typography>
              {metric.status_reason && <Typography variant="body2" color="warning.main">{metric.status_reason}</Typography>}
            </Box>
            <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center">
              <Chip size="small" label={`n=${metric.denominator ?? '—'}`} />
              <Chip size="small" label={metric.evidence_mode} variant="outlined" />
              <Link href={metric.reference_url} target="_blank" rel="noreferrer">{metric.standard}</Link>
            </Stack>
          </Stack>
        </Paper>)}
      </Stack>
    </AccordionDetails>
  </Accordion>
}

export function EvaluationPage() {
  const query = useQuery({ queryKey: ['evaluation-latest'], queryFn: api.latestEvaluation, retry: false })
  const report = query.data
  const headline = useMemo(() => report ? [
    ['Agent Task Success', suiteMetric(report, 'agent', 'task_success_rate')],
    ['Router Macro-F1', suiteMetric(report, 'agent', 'router_macro_f1')],
    ['Tool Executable Accuracy', suiteMetric(report, 'tool', 'executable_accuracy')],
    ['RAG Recall@20', suiteMetric(report, 'rag', 'recall_at_20')],
    ['Scientific Simple Regret', suiteMetric(report, 'scientific', 'offline_replay_simple_regret')],
    ['Memory Recall@5', suiteMetric(report, 'memory', 'recall_at_5')],
  ] as Array<[string, EvaluationMetric | undefined]> : [], [report])
  if (query.isLoading) return <Box minHeight={420} display="grid" sx={{ placeItems: 'center' }}><CircularProgress /></Box>
  if (query.error || !report) {
    const detail = query.error instanceof ApiError && query.error.status === 404
      ? '尚无已完成报告。请先在 CLI 运行 python scripts/run_eval_showcase.py。'
      : '评估报告加载失败。'
    return <Alert severity="warning">{detail}</Alert>
  }
  const backendLooksFake = [report.environment.embedding_backend, report.environment.reranker_backend]
    .some(value => /fake|disabled|lexical/i.test(value || ''))
  return <>
    <PageHeader
      eyebrow="Standards-based evidence"
      title="评估中心"
      description="公认指标、可复现配置、置信区间与组件消融；不展示自定义综合能力分。"
      action={<Stack direction="row" gap={1}>
        <Button startIcon={<PrintRounded />} variant="outlined" onClick={() => window.print()}>打印 / PDF</Button>
        <Button
          startIcon={<DownloadRounded />}
          variant="contained"
          href={`/api/v2/evaluations/${encodeURIComponent(report.report_id)}/download`}
        >下载 JSON</Button>
      </Stack>}
    />
    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Stack direction={{ xs: 'column', lg: 'row' }} justifyContent="space-between" gap={2}>
        <Box>
          <Typography fontWeight={800}>{report.report_id}</Typography>
          <Typography variant="body2" color="text.secondary">
            {new Date(report.generated_at).toLocaleString()} · commit {report.git_commit.slice(0, 12)} · schema {report.schema_version}
          </Typography>
        </Box>
        <Stack direction="row" gap={1} flexWrap="wrap">
          <Chip icon={<VerifiedRounded />} label={`Embedding: ${report.environment.embedding_backend}${suiteByKey(report, 'rag')?.status === 'completed' ? '' : ' · not used'}`} color={backendLooksFake ? 'warning' : 'success'} />
          <Chip label={`Reranker: ${report.environment.reranker_backend}${suiteByKey(report, 'rag')?.status === 'completed' ? '' : ' · not used'}`} color={backendLooksFake ? 'warning' : 'success'} />
          <Chip label={report.environment.judge_enabled ? `Judge: ${report.environment.judge_model || 'enabled'}` : 'Judge: NOT RUN'} variant="outlined" />
        </Stack>
      </Stack>
      {backendLooksFake && <Alert severity="warning" sx={{ mt: 2 }}>当前报告包含 fake / disabled / lexical fallback backend；这些结果不得描述为真实模型质量。</Alert>}
      <Typography variant="caption" color="text.secondary" display="block" mt={1}>
        数据集：{Object.entries(report.dataset_versions).map(([key, value]) => `${key}=${value}`).join(' · ')}
      </Typography>
    </Paper>

    <Grid container spacing={2} mb={3}>
      {headline.map(([label, metric]) => <Grid key={label} size={{ xs: 12, sm: 6, lg: 4 }}><MetricTile label={label} metric={metric} /></Grid>)}
    </Grid>

    <Grid container spacing={2} mb={3}>
      <Grid size={{ xs: 12, xl: 7 }}><Card><CardContent><Typography variant="h2" mb={1}>RAG 检索消融</Typography><RagAblationChart report={report} /></CardContent></Card></Grid>
      <Grid size={{ xs: 12, xl: 5 }}><Card><CardContent><Typography variant="h2" mb={1}>Agent Router Confusion Matrix</Typography><ConfusionMatrix report={report} /></CardContent></Card></Grid>
      <Grid size={{ xs: 12, xl: 7 }}><Card><CardContent><Typography variant="h2" mb={1}>Scientific 与 Context 消融</Typography><ComparisonChart report={report} /></CardContent></Card></Grid>
      <Grid size={{ xs: 12, xl: 5 }}><Card><CardContent><Typography variant="h2" mb={1}>失败切片</Typography><FailureSlices report={report} /></CardContent></Card></Grid>
    </Grid>

    <Typography variant="h2" mb={1.5}>指标定义与证据</Typography>
    <Stack spacing={1}>{report.suites.map(suite => <MetricDetails key={suite.key} suite={suite} />)}</Stack>
  </>
}
