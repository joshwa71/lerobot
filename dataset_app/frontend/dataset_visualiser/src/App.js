import React from 'react';
import {
    BrowserRouter as Router,
    Routes,
    Route,
    useParams,
    useNavigate
} from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import './App.css';
import DatasetListPage from './components/DatasetListPage';
import EpisodeViewPage from './components/EpisodeViewPage';

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#1976d2',
        },
        secondary: {
            main: '#dc004e',
        },
        background: {
            default: '#f5f5f5',
        },
    },
    typography: {
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
        h3: {
            fontWeight: 600,
        },
        h6: {
            fontWeight: 500,
        },
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 12,
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    textTransform: 'none',
                },
            },
        },
    },
});

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
                <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
                    <Routes>
                        <Route path="/" element={<DatasetListPage />} />
                        <Route path="/:repoId" element={<DatasetRedirect />} />
                        <Route path="/:repoId/episode/:episodeId" element={<EpisodeViewPage />} />
                    </Routes>
                </Box>
            </Router>
        </ThemeProvider>
    );
}

// Helper component to redirect from /:repoId to /:repoId/episode/0
function DatasetRedirect() {
    const { repoId } = useParams();
    const navigate = useNavigate();
    React.useEffect(() => {
        navigate(`/${repoId}/episode/0`);
    }, [repoId, navigate]);
    return <Box display="flex" justifyContent="center" alignItems="center" height="100vh">Loading dataset...</Box>; 
}

export default App;
