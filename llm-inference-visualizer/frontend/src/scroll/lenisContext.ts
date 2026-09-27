import { createContext, useContext } from 'react'
import type Lenis from 'lenis'

export const LenisContext = createContext<Lenis | null>(null)
export const PageProgressContext = createContext(0)
export const usePageProgress = () => useContext(PageProgressContext)
