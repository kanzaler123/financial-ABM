import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { LanguageContext, type Language } from './i18n'

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => localStorage.getItem('abm-language') === 'en' ? 'en' : 'zh')
  useEffect(() => {
    localStorage.setItem('abm-language', language)
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en'
    document.title = language === 'zh' ? 'ABM · 金融市场仿真实验室' : 'ABM · Financial Market Laboratory'
  }, [language])
  const value = useMemo(() => ({
    language, setLanguage,
    t: (english: string, chinese: string) => language === 'zh' ? chinese : english,
    format: (number: number, digits = 4) => Number.isFinite(number)
      ? number.toLocaleString(language === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: digits }) : '—',
  }), [language])
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}
