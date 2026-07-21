import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#0f766e', dark: '#115e59', light: '#ccfbf1' },
    secondary: { main: '#475569' },
    success: { main: '#15803d' },
    warning: { main: '#b45309' },
    error: { main: '#b91c1c' },
    background: { default: '#f5f7f7', paper: '#ffffff' },
    text: { primary: '#172321', secondary: '#5e6d69' },
    divider: '#dfe7e4',
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: 'Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif',
    h1: { fontSize: '1.75rem', fontWeight: 750, letterSpacing: '-0.02em' },
    h2: { fontSize: '1.35rem', fontWeight: 700 },
    h3: { fontSize: '1.05rem', fontWeight: 700 },
    button: { textTransform: 'none', fontWeight: 650 },
  },
  components: {
    MuiCard: { styleOverrides: { root: { border: '1px solid #dfe7e4', boxShadow: '0 1px 2px rgba(15, 31, 28, .035)' } } },
    MuiButton: { defaultProps: { disableElevation: true } },
    MuiChip: { styleOverrides: { root: { fontWeight: 650 } } },
  },
})
