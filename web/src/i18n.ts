import { createContext, useContext } from 'react'

export type Language = 'zh' | 'en'
type Translator = (english: string, chinese: string) => string
interface LanguageContextValue {
  language: Language
  setLanguage: (language: Language) => void
  t: Translator
  format: (value: number, digits?: number) => string
}

export const LanguageContext = createContext<LanguageContextValue>(null!)

export function useLanguage() { return useContext(LanguageContext) }
