import * as echarts from 'echarts'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLanguage } from './i18n'

export interface SeriesSpec {
  name: string
  field: string
  color: string
  type?: 'line' | 'bar'
  dashed?: boolean
  secondary?: boolean
}

interface ChartProps {
  title: string
  rows: Array<Record<string, number>>
  series: SeriesSpec[]
  yLabel?: string
  valueFormatter?: (value: number) => string
  secondaryLabel?: string
  toolbar?: ReactNode
}

export function Chart({ title, rows, series, yLabel, valueFormatter, secondaryLabel, toolbar }: ChartProps) {
  const { t, format } = useLanguage()
  const root = useRef<HTMLDivElement>(null)
  const [showTable, setShowTable] = useState(false)
  const option = useMemo<echarts.EChartsOption>(() => ({
    animation: false,
    color: series.map((item) => item.color),
    grid: { left: 60, right: secondaryLabel ? 58 : 24, top: 64, bottom: 48 },
    legend: series.length > 1 ? { top: 12, icon: 'roundRect', itemWidth: 12, itemHeight: 7, itemGap: 18, textStyle: { color: '#7085a9', fontSize: 11 } } : undefined,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      valueFormatter: (value) => valueFormatter ? valueFormatter(Number(value)) : format(Number(value)),
    },
    xAxis: {
      type: 'category',
      data: rows.map((row) => row.day),
      name: t('Simulation day', '仿真天数'),
      nameLocation: 'middle',
      nameGap: 30,
      nameTextStyle: { color: '#7085a9' },
      boundaryGap: false,
      axisLine: { lineStyle: { color: '#dce6f7' } },
      axisLabel: { color: '#7085a9' },
      axisTick: { show: false },
    },
    yAxis: [{
      type: 'value',
      name: yLabel,
      scale: true,
      nameTextStyle: { color: '#7085a9' },
      axisLabel: { color: '#7085a9' },
      splitLine: { lineStyle: { color: 'rgba(144, 171, 216, .15)', width: 1 } },
    }, ...(secondaryLabel ? [{ type: 'value' as const, name: secondaryLabel, min: 0,
      max: (extent: { max: number }) => extent.max * 4,
      nameTextStyle: { color: '#7085a9' }, axisLabel: { color: '#7085a9', formatter: (value: number) => format(value, 0) }, splitLine: { show: false },
    }] : [])],
    series: series.map((item) => ({
      name: item.name,
      type: item.type ?? 'line',
      yAxisIndex: item.secondary ? 1 : 0,
      data: rows.map((row) => row[item.field]),
      showSymbol: false,
      lineStyle: { width: 2, type: item.dashed ? 'dashed' : 'solid' },
      itemStyle: { opacity: item.type === 'bar' ? .55 : 1, borderRadius: item.type === 'bar' ? [2, 2, 0, 0] : undefined },
      z: item.type === 'bar' ? 0 : 2,
      emphasis: { focus: 'series' },
    })),
  }), [rows, series, valueFormatter, yLabel, secondaryLabel, t, format])

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
  }, [option, showTable])

  return (
    <section className="chart-card">
      <header>
        <div>
          <h3>{title}</h3>
          {!toolbar && <p>{format(rows.length, 0)} {t('observations', '条观测')}</p>}
        </div>
        <div className="chart-tools">{toolbar}<button className="ghost" type="button" onClick={() => setShowTable((value) => !value)}>
          {showTable ? t('Chart', '图表') : t('Table', '数据')}
        </button></div>
      </header>
      {showTable ? (
        <div key="table" className="table-scroll" tabIndex={0}>
          <table>
            <thead><tr><th>{t('Day', '天数')}</th>{series.map((item) => <th key={item.field}>{item.name}</th>)}</tr></thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.day}><td>{format(row.day, 0)}</td>{series.map((item) => <td key={item.field}>{valueFormatter?.(row[item.field]) ?? format(row[item.field], 6)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div key="chart" ref={root} className="chart" role="img" aria-label={`${title} · ${t('time-series chart', '时间序列图')}`} />}
    </section>
  )
}
