import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
    Container,
    Typography,
    Grid,
    Card,
    CardContent,
    CardActionArea,
    Chip,
    Box,
    CircularProgress,
    Alert,
    Paper,
    Divider,
    Checkbox,
    FormControlLabel,
    TextField,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    LinearProgress,
    Snackbar
} from '@mui/material';
import {
    Dataset as DatasetIcon,
    Movie as VideoIcon,
    Timer as TimerIcon,
    Analytics as AnalyticsIcon,
    MergeType as MergeIcon,
    Check as CheckIcon,
    Error as ErrorIcon
} from '@mui/icons-material';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

function DatasetListPage() {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedDatasets, setSelectedDatasets] = useState([]);
    const [newDatasetName, setNewDatasetName] = useState('');
    const [mergeDialogOpen, setMergeDialogOpen] = useState(false);
    const [merging, setMerging] = useState(false);
    const [mergeOperationId, setMergeOperationId] = useState(null);
    const [mergeStatus, setMergeStatus] = useState(null);
    const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
    const navigate = useNavigate();

    useEffect(() => {
        loadDatasets();
    }, []);

    // Poll for merge status updates
    useEffect(() => {
        let interval;
        if (mergeOperationId && merging) {
            interval = setInterval(async () => {
                try {
                    const response = await axios.get(`${API_URL}/api/datasets/merge/${mergeOperationId}/status`);
                    setMergeStatus(response.data);
                    
                    if (response.data.status === 'completed') {
                        setMerging(false);
                        setMergeDialogOpen(false);
                        setSelectedDatasets([]);
                        setNewDatasetName('');
                        setSnackbar({
                            open: true,
                            message: 'Datasets merged successfully!',
                            severity: 'success'
                        });
                        // Reload datasets to show the new merged dataset
                        loadDatasets();
                    } else if (response.data.status === 'error') {
                        setMerging(false);
                        setSnackbar({
                            open: true,
                            message: `Merge failed: ${response.data.message}`,
                            severity: 'error'
                        });
                    }
                } catch (err) {
                    console.error('Error checking merge status:', err);
                }
            }, 2000); // Poll every 2 seconds
        }
        
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [mergeOperationId, merging]);

    const loadDatasets = () => {
        setLoading(true);
        axios.get(`${API_URL}/api/datasets`)
            .then(response => {
                setDatasets(response.data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error fetching datasets:", err);
                setError('Failed to load datasets. Ensure the backend is running and LEROBOT_ROOT_PATH is correctly set.');
                setLoading(false);
            });
    };

    const handleDatasetClick = (repoId) => {
        navigate(`/${repoId}/episode/0`);
    };

    const handleDatasetSelection = (repoId, checked) => {
        if (checked) {
            if (selectedDatasets.length < 2) {
                setSelectedDatasets([...selectedDatasets, repoId]);
            }
        } else {
            setSelectedDatasets(selectedDatasets.filter(id => id !== repoId));
        }
    };

    const handleMergeClick = () => {
        if (selectedDatasets.length === 2) {
            setMergeDialogOpen(true);
        }
    };

    const handleMergeConfirm = async () => {
        if (!newDatasetName.trim()) {
            setSnackbar({
                open: true,
                message: 'Please enter a name for the merged dataset',
                severity: 'warning'
            });
            return;
        }

        try {
            setMerging(true);
            const response = await axios.post(`${API_URL}/api/datasets/merge`, {
                source1: selectedDatasets[0],
                source2: selectedDatasets[1],
                target_name: newDatasetName.trim()
            });
            
            setMergeOperationId(response.data.operation_id);
            setMergeStatus({
                status: 'started',
                message: 'Starting merge operation...'
            });
            
        } catch (err) {
            setMerging(false);
            setSnackbar({
                open: true,
                message: `Failed to start merge: ${err.response?.data?.error || err.message}`,
                severity: 'error'
            });
        }
    };

    const handleMergeCancel = () => {
        setMergeDialogOpen(false);
        setNewDatasetName('');
    };

    if (loading) {
        return (
            <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
                <CircularProgress size={60} />
            </Container>
        );
    }

    if (error) {
        return (
            <Container maxWidth="lg" sx={{ mt: 4 }}>
                <Alert severity="error">{error}</Alert>
            </Container>
        );
    }

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'grey.50' }}>
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <DatasetIcon color="primary" sx={{ fontSize: 40 }} />
                    <Typography variant="h3" component="h1" fontWeight="bold">
                        LeRobot Datasets
                    </Typography>
                </Box>
                <Typography variant="body1" color="text.secondary">
                    Browse and visualize your locally stored LeRobot datasets. Click on any dataset to start exploring its episodes.
                </Typography>
            </Paper>

            {/* Merge Controls */}
            {datasets.length > 1 && (
                <Paper elevation={1} sx={{ p: 3, mb: 4 }}>
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                        <MergeIcon color="secondary" />
                        <Typography variant="h6" fontWeight="bold">
                            Merge Datasets
                        </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" mb={2}>
                        Select exactly 2 datasets to merge them into a new combined dataset.
                    </Typography>
                    <Box display="flex" alignItems="center" gap={2}>
                        <Chip 
                            label={`${selectedDatasets.length}/2 selected`}
                            color={selectedDatasets.length === 2 ? "success" : "default"}
                            variant="outlined"
                        />
                        <Button
                            variant="contained"
                            startIcon={<MergeIcon />}
                            onClick={handleMergeClick}
                            disabled={selectedDatasets.length !== 2}
                        >
                            Merge Selected Datasets
                        </Button>
                    </Box>
                </Paper>
            )}

            {datasets.length > 0 ? (
                <Grid container spacing={3}>
                    {datasets.map(dataset => (
                        <Grid item xs={12} sm={6} md={4} key={dataset.repo_id}>
                            <Card 
                                elevation={2}
                                sx={{ 
                                    height: '100%',
                                    transition: 'all 0.3s ease',
                                    '&:hover': {
                                        elevation: 8,
                                        transform: 'translateY(-4px)'
                                    },
                                    border: selectedDatasets.includes(dataset.repo_id) ? '2px solid' : 'none',
                                    borderColor: 'primary.main'
                                }}
                            >
                                <Box sx={{ position: 'relative' }}>
                                    {datasets.length > 1 && (
                                        <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 1 }}>
                                            <FormControlLabel
                                                control={
                                                    <Checkbox
                                                        checked={selectedDatasets.includes(dataset.repo_id)}
                                                        onChange={(e) => handleDatasetSelection(dataset.repo_id, e.target.checked)}
                                                        disabled={!selectedDatasets.includes(dataset.repo_id) && selectedDatasets.length >= 2}
                                                        onClick={(e) => e.stopPropagation()}
                                                    />
                                                }
                                                label=""
                                                sx={{ m: 0 }}
                                            />
                                        </Box>
                                    )}
                                    <CardActionArea 
                                        onClick={() => handleDatasetClick(dataset.repo_id)}
                                        sx={{ height: '100%', p: 0 }}
                                    >
                                        <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', pt: datasets.length > 1 ? 6 : 2 }}>
                                            <Box display="flex" alignItems="center" gap={1} mb={2}>
                                                <VideoIcon color="primary" />
                                                <Typography variant="h6" component="h2" fontWeight="bold" noWrap>
                                                    {dataset.repo_id}
                                                </Typography>
                                            </Box>
                                            
                                            <Divider sx={{ mb: 2 }} />
                                            
                                            <Box display="flex" flexDirection="column" gap={1.5} flexGrow={1}>
                                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                                    <Typography variant="body2" color="text.secondary">
                                                        Episodes
                                                    </Typography>
                                                    <Chip 
                                                        label={dataset.total_episodes} 
                                                        size="small" 
                                                        variant="outlined"
                                                        icon={<AnalyticsIcon />}
                                                    />
                                                </Box>
                                                
                                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                                    <Typography variant="body2" color="text.secondary">
                                                        Total Frames
                                                    </Typography>
                                                    <Chip 
                                                        label={dataset.total_frames.toLocaleString()} 
                                                        size="small" 
                                                        variant="outlined"
                                                    />
                                                </Box>
                                                
                                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                                    <Typography variant="body2" color="text.secondary">
                                                        FPS
                                                    </Typography>
                                                    <Chip 
                                                        label={`${dataset.fps} fps`} 
                                                        size="small" 
                                                        variant="outlined"
                                                        icon={<TimerIcon />}
                                                    />
                                                </Box>
                                            </Box>
                                            
                                            <Box mt={2}>
                                                <Chip 
                                                    label={`v${dataset.codebase_version}`} 
                                                    size="small" 
                                                    color="secondary"
                                                    variant="filled"
                                                />
                                            </Box>
                                        </CardContent>
                                    </CardActionArea>
                                </Box>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            ) : (
                <Paper elevation={1} sx={{ p: 4, textAlign: 'center' }}>
                    <DatasetIcon sx={{ fontSize: 60, color: 'grey.400', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                        No datasets found
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Make sure the backend is configured with the correct `lerobot_root_path` and that datasets exist in that location.
                    </Typography>
                </Paper>
            )}

            {/* Merge Dialog */}
            <Dialog open={mergeDialogOpen} onClose={handleMergeCancel} maxWidth="sm" fullWidth>
                <DialogTitle>
                    <Box display="flex" alignItems="center" gap={1}>
                        <MergeIcon />
                        Merge Datasets
                    </Box>
                </DialogTitle>
                <DialogContent>
                    {!merging ? (
                        <>
                            <Typography variant="body1" gutterBottom>
                                You are about to merge the following datasets:
                            </Typography>
                            <Box my={2}>
                                <Chip label={selectedDatasets[0]} variant="outlined" sx={{ mr: 1, mb: 1 }} />
                                <Typography variant="body2" component="span" sx={{ mx: 1 }}>+</Typography>
                                <Chip label={selectedDatasets[1]} variant="outlined" sx={{ mb: 1 }} />
                                <Typography variant="body2" component="span" sx={{ mx: 1 }}>=</Typography>
                                <Chip label={newDatasetName || 'New Dataset'} color="primary" sx={{ mb: 1 }} />
                            </Box>
                            <TextField
                                fullWidth
                                label="New Dataset Name"
                                value={newDatasetName}
                                onChange={(e) => setNewDatasetName(e.target.value)}
                                placeholder="Enter name for merged dataset"
                                helperText="The new dataset will be created in your outputs directory"
                                variant="outlined"
                                margin="normal"
                            />
                        </>
                    ) : (
                        <>
                            <Box display="flex" alignItems="center" gap={2} mb={2}>
                                <CircularProgress size={24} />
                                <Typography variant="h6">Merging Datasets...</Typography>
                            </Box>
                            {mergeStatus && (
                                <>
                                    <Typography variant="body2" color="text.secondary" mb={2}>
                                        {mergeStatus.message}
                                    </Typography>
                                    <LinearProgress />
                                </>
                            )}
                        </>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleMergeCancel} disabled={merging}>
                        Cancel
                    </Button>
                    <Button 
                        onClick={handleMergeConfirm} 
                        variant="contained" 
                        disabled={merging || !newDatasetName.trim()}
                        startIcon={merging ? <CircularProgress size={16} /> : <CheckIcon />}
                    >
                        {merging ? 'Merging...' : 'Confirm Merge'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Snackbar for notifications */}
            <Snackbar
                open={snackbar.open}
                autoHideDuration={6000}
                onClose={() => setSnackbar({ ...snackbar, open: false })}
            >
                <Alert 
                    onClose={() => setSnackbar({ ...snackbar, open: false })} 
                    severity={snackbar.severity}
                    sx={{ width: '100%' }}
                >
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </Container>
    );
}

export default DatasetListPage; 