import * as echarts from 'echarts'
import { useEffect, useMemo, useRef, useState } from 'react'

export interface SeriesSpec {
  name: string
  field: string
  color: string
}

interface ChartProps {
  title: string
  rows: Array<Record<string, number>>
  series: SeriesSpec[]
  yLabel?: string
  valueFormatter?: (value: number) => string
}

export function Chart({ title, rows, series, yLabel, valueFormatter }: ChartProps) {
  const root = useRef<HTMLDivElement>(null)
  const [showTable, setShowTable] = useState(false)
  const option = useMemo<echarts.EChartsOption>(() => ({
    animation: false,
    color: series.map((item) => item.color),
    grid: { left: 64, right: 24, top: 52, bottom: 42 },
    legend: series.length > 1 ? { top: 10, textStyle: { color: 'var(--text-secondary)' } } : undefined,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      valueFormatter: valueFormatter ? (value) => valueFormatter(Number(value)) : undefined,
    },
    xAxis: {
      type: 'category',
      data: rows.map((row) => row.day),
      name: 'Day',
      boundaryGap: false,
      axisLine: { lineStyle: { color: '#898781' } },
    },
    yAxis: {
      type: 'value',
      name: yLabel,
      scale: true,
      splitLine: { lineStyle: { color: '#e1e0d9', width: 1 } },
    },
    series: series.map((item) => ({
      name: item.name,
      type: 'line',
      data: rows.map((row) => row[item.field]),
      showSymbol: false,
      lineStyle: { width: 2 },
      emphasis: { focus: 'series' },
    })),
  }), [rows, series, valueFormatter, yLabel])

  useEffect(() => {
    if (!root.current) return
    const chart = echarts.init(root.current)
    chart.setOption(option)
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(root.current)
    return () => {
      resize.disconnect()
      chart.dispose()
    }
  }, [option])

  return (
    <section className="chart-card">
      <header>
        <div>
          <h3>{title}</h3>
          <p>{rows.length.toLocaleString()} observations</p>
        </div>
        <button className="ghost" type="button" onClick={() => setShowTable((value) => !value)}>
          {showTable ? 'Show chart' : 'View table'}
        </button>
      </header>
      {showTable ? (
        <div className="table-scroll" tabIndex={0}>
          <table>
            <thead><tr><th>Day</th>{series.map((item) => <th key={item.field}>{item.name}</th>)}</tr></thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.day}><td>{row.day}</td>{series.map((item) => <td key={item.field}>{valueFormatter?.(row[item.field]) ?? row[item.field]?.toPrecision(6)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div ref={root} className="chart" role="img" aria-label={`${title} time-series chart`} />}
    </section>
  )
}
