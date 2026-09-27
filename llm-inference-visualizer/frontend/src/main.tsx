import React from 'react'
import ReactDOM from 'react-dom/client'
import '@fontsource/instrument-sans/400.css'
import '@fontsource/instrument-sans/600.css'
import '@fontsource/instrument-serif/400.css'
import '@fontsource/instrument-serif/400-italic.css'
import App from './App'
import './style.css'
import './responsive.css'
import { LenisProvider } from './scroll/LenisProvider'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><LenisProvider><App /></LenisProvider></React.StrictMode>,
)
