import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import Dygraph from 'dygraphs';
import 'dygraphs/dist/dygraph.min.css';
import {
    Box,
    Grid,
    Paper,
    Typography,
    IconButton,
    Slider,
    Card,
    CardContent,
    Drawer,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Divider,
    Chip,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    FormGroup,
    FormControlLabel,
    Checkbox,
    Toolbar,
    Alert,
    CircularProgress,
    AppBar,
    Button,
    Container,
    Stack
} from '@mui/material';
import {
    PlayArrow as PlayIcon,
    Pause as PauseIcon,
    SkipPrevious as PrevIcon,
    SkipNext as NextIcon,
    Replay as ReplayIcon,
    ArrowBack as BackIcon,
    Timeline as TimelineIcon,
    VideoLibrary as VideoIcon,
    BarChart as ChartIcon,
    Settings as SettingsIcon
} from '@mui/icons-material';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const DRAWER_WIDTH = 320;

const formatTime = (timeInSeconds) => {
    const M = Math.floor(timeInSeconds / 60);
    const S = Math.floor(timeInSeconds % 60);
    const MS = Math.floor((timeInSeconds % 1) * 1000);
    return `${String(M).padStart(2, '0')}:${String(S).padStart(2, '0')}.${String(MS).padStart(3, '0')}`;
};

function EpisodeViewPage() {
    const { repoId, episodeId } = useParams();
    const navigate = useNavigate();
    const [episodeData, setEpisodeData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [currentVideoTime, setCurrentVideoTime] = useState(0);
    const [selectedVideoStreams, setSelectedVideoStreams] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [sliderValue, setSliderValue] = useState(0);
    const [isUserInteracting, setIsUserInteracting] = useState(false);

    const dygraphRef = useRef(null);
    const dygraphInstanceRef = useRef(null);
    const videoRefs = useRef({});
    const lastUpdateTimeRef = useRef(0);

    const updateDygraphSelection = useCallback((time) => {
        if (dygraphInstanceRef.current && episodeData?.csv_data && !isUserInteracting) {
            const data = dygraphInstanceRef.current.rawData_;
            if (!data || data.length === 0) return;

            let closestRow = -1;
            let minDiff = Infinity;
            for (let i = 0; i < data.length; i++) {
                const rowTime = data[i][0];
                const diff = Math.abs(rowTime - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestRow = i;
                }
            }
            if (closestRow !== -1) {
                dygraphInstanceRef.current.setSelection(closestRow);
            }
        }
    }, [episodeData?.csv_data, isUserInteracting]);

    useEffect(() => {
        setLoading(true);
        axios.get(`${API_URL}/api/datasets/${repoId}/episodes/${episodeId}/data`)
            .then(response => {
                setEpisodeData(response.data);
                if (response.data.videos_info) {
                    setSelectedVideoStreams(response.data.videos_info.map(v => v.filename));
                }
                setLoading(false);
            })
            .catch(err => {
                console.error("Error fetching episode data:", err);
                setError('Failed to load episode data.');
                setLoading(false);
            });

        return () => {
            if (dygraphInstanceRef.current) {
                dygraphInstanceRef.current.destroy();
                dygraphInstanceRef.current = null;
            }
        };
    }, [repoId, episodeId]);

    useEffect(() => {
        if (episodeData && episodeData.csv_data && dygraphRef.current) {
            if (dygraphInstanceRef.current) {
                dygraphInstanceRef.current.destroy();
            }
            
            const g = new Dygraph(
                dygraphRef.current,
                episodeData.csv_data,
                {
                    legend: 'always',
                    labelsSeparateLines: true,
                    ylabel: 'Value',
                    xlabel: 'Time (s)',
                    showRangeSelector: true,
                    rangeSelectorHeight: 30,
                    interactionModel: Dygraph.defaultInteractionModel,
                    highlightCallback: (event, x, points, row, seriesName) => {
                        if (points && points.length > 0 && !isUserInteracting) {
                            const time = points[0].xval;
                            // Throttle updates to prevent excessive calls
                            const now = Date.now();
                            if (now - lastUpdateTimeRef.current > 50) { // Max 20 updates per second
                                lastUpdateTimeRef.current = now;
                                setCurrentVideoTime(time);
                                
                                Object.values(videoRefs.current).forEach(video => {
                                    if (video && video.readyState >= 3) {
                                        if (Math.abs(video.currentTime - time) > 0.1) {
                                            video.currentTime = time;
                                        }
                                    }
                                });
                            }
                        }
                    },
                    clickCallback: (event, x, points) => {
                        if (points && points.length > 0) {
                            const time = points[0].xval;
                            setCurrentVideoTime(time);
                            Object.values(videoRefs.current).forEach(video => {
                                if (video) video.currentTime = time;
                            });
                        }
                    }
                }
            );
            dygraphInstanceRef.current = g;
        }
    }, [episodeData, isUserInteracting]);

    const handlePlayPause = () => {
        const newPlayState = !isPlaying;
        setIsPlaying(newPlayState);
        Object.values(videoRefs.current).forEach(video => {
            if (video) {
                newPlayState ? video.play() : video.pause();
            }
        });
    };

    const handleSliderChange = (event, newValue) => {
        setIsUserInteracting(true);
        setSliderValue(newValue);
        const primaryVideo = videoRefs.current[selectedVideoStreams[0]];
        if (primaryVideo && primaryVideo.duration) {
            const newTime = (newValue / 100) * primaryVideo.duration;
            setCurrentVideoTime(newTime);
            Object.values(videoRefs.current).forEach(video => {
                if (video) video.currentTime = newTime;
            });
            updateDygraphSelection(newTime);
        }
    };

    const handleSliderChangeCommitted = () => {
        setTimeout(() => setIsUserInteracting(false), 100);
    };

    const handleVideoTimeUpdate = (e) => {
        if (isUserInteracting) return;
        
        const primaryVideoKey = selectedVideoStreams.length > 0 ? selectedVideoStreams[0] : null;
        if (primaryVideoKey && e.target === videoRefs.current[primaryVideoKey]) {
            const time = e.target.currentTime;
            const duration = e.target.duration;
            
            setCurrentVideoTime(time);
            if (duration) {
                const newSliderValue = (time / duration) * 100;
                setSliderValue(newSliderValue);
            }
            updateDygraphSelection(time);
        }
    };

    const toggleVideoStream = (filename) => {
        setSelectedVideoStreams(prev => 
            prev.includes(filename) ? prev.filter(f => f !== filename) : [...prev, filename]
        );
    };
    
    const navigateEpisode = (offset) => {
        const newEpisodeId = parseInt(episodeId) + offset;
        if (episodeData && newEpisodeId >= 0 && newEpisodeId < episodeData.total_episodes_in_dataset) {
            navigate(`/${repoId}/episode/${newEpisodeId}`);
        }
    };

    const jumpToTime = (seconds) => {
        Object.values(videoRefs.current).forEach(video => {
            if (video) video.currentTime = Math.max(0, video.currentTime + seconds);
        });
    };

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
                <CircularProgress size={60} />
            </Box>
        );
    }

    if (error) {
        return (
            <Container maxWidth="lg" sx={{ mt: 4 }}>
                <Alert severity="error">{error}</Alert>
            </Container>
        );
    }

    if (!episodeData) {
        return (
            <Container maxWidth="lg" sx={{ mt: 4 }}>
                <Alert severity="warning">No episode data available.</Alert>
            </Container>
        );
    }

    const mainVideoDuration = videoRefs.current[selectedVideoStreams[0]]?.duration || 0;

    return (
        <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'grey.50' }}>
            {/* Sidebar */}
            <Drawer
                variant="permanent"
                sx={{
                    width: DRAWER_WIDTH,
                    flexShrink: 0,
                    '& .MuiDrawer-paper': {
                        width: DRAWER_WIDTH,
                        boxSizing: 'border-box',
                        bgcolor: 'background.paper'
                    },
                }}
            >
                <Toolbar>
                    <Button
                        startIcon={<BackIcon />}
                        onClick={() => navigate('/')}
                        variant="outlined"
                        size="small"
                    >
                        Back to Datasets
                    </Button>
                </Toolbar>
                
                <Divider />
                
                <Box sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                        {repoId}
                    </Typography>
                    <Chip label={`Episode ${episodeId}`} color="primary" />
                </Box>
                
                <Divider />
                
                <List>
                    <ListItem>
                        <ListItemIcon><TimelineIcon /></ListItemIcon>
                        <ListItemText 
                            primary="FPS" 
                            secondary={episodeData.fps}
                        />
                    </ListItem>
                    <ListItem>
                        <ListItemIcon><VideoIcon /></ListItemIcon>
                        <ListItemText 
                            primary="Total Episodes" 
                            secondary={episodeData.total_episodes_in_dataset}
                        />
                    </ListItem>
                </List>
                
                <Divider />
                
                <Box sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                        Episode Navigation
                    </Typography>
                    <Stack direction="row" spacing={1} mb={2}>
                        <Button 
                            onClick={() => navigateEpisode(-1)} 
                            disabled={parseInt(episodeId) === 0}
                            size="small"
                            variant="outlined"
                        >
                            Previous
                        </Button>
                        <Button 
                            onClick={() => navigateEpisode(1)} 
                            disabled={parseInt(episodeId) === episodeData.total_episodes_in_dataset - 1}
                            size="small"
                            variant="outlined"
                        >
                            Next
                        </Button>
                    </Stack>
                    
                    <FormControl fullWidth size="small">
                        <InputLabel>Episode</InputLabel>
                        <Select
                            value={episodeId}
                            label="Episode"
                            onChange={(e) => navigate(`/${repoId}/episode/${e.target.value}`)}
                        >
                            {episodeData.all_episode_indices.map(idx => (
                                <MenuItem key={idx} value={idx}>Episode {idx}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Box>
                
                {episodeData.videos_info && episodeData.videos_info.length > 0 && (
                    <>
                        <Divider />
                        <Box sx={{ p: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                                Video Streams
                            </Typography>
                            <FormGroup>
                                {episodeData.videos_info.map(video => (
                                    <FormControlLabel
                                        key={video.filename}
                                        control={
                                            <Checkbox
                                                checked={selectedVideoStreams.includes(video.filename)}
                                                onChange={() => toggleVideoStream(video.filename)}
                                            />
                                        }
                                        label={video.filename}
                                    />
                                ))}
                            </FormGroup>
                        </Box>
                    </>
                )}
            </Drawer>

            {/* Main Content */}
            <Box component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Top Bar */}
                <AppBar position="static" color="default" elevation={1}>
                    <Toolbar>
                        <Typography variant="h6" sx={{ flexGrow: 1 }}>
                            Episode {episodeId} Visualization
                        </Typography>
                        {episodeData.language_instruction && (
                            <Chip 
                                label={`Instruction: ${episodeData.language_instruction}`}
                                variant="outlined"
                                sx={{ maxWidth: 400 }}
                            />
                        )}
                    </Toolbar>
                </AppBar>

                {/* Content Grid */}
                <Box sx={{ flexGrow: 1, p: 2, overflow: 'auto' }}>
                    <Grid container spacing={2} sx={{ height: '100%' }}>
                        {/* Videos Section */}
                        <Grid item xs={12} lg={6}>
                            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                                <Typography variant="h6" gutterBottom>
                                    Video Streams
                                </Typography>
                                <Box sx={{ 
                                    flexGrow: 1, 
                                    display: 'flex', 
                                    flexDirection: 'column', 
                                    gap: 2, 
                                    overflow: 'auto'
                                }}>
                                    {episodeData.videos_info?.filter(v => selectedVideoStreams.includes(v.filename)).map(video => (
                                        <Card key={video.filename} sx={{ width: '100%' }}>
                                            <CardContent sx={{ p: 1 }}>
                                                <Typography variant="caption" gutterBottom>
                                                    {video.filename}
                                                </Typography>
                                                <video 
                                                    ref={el => videoRefs.current[video.filename] = el} 
                                                    muted 
                                                    loop 
                                                    controls={false}
                                                    onTimeUpdate={handleVideoTimeUpdate}
                                                    style={{ width: '100%', height: 'auto', maxHeight: 300 }}
                                                    src={`${API_URL}${video.url}`}
                                                >
                                                    Your browser does not support the video tag.
                                                </video>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </Box>
                            </Paper>
                        </Grid>
                        
                        {/* Chart Section */}
                        <Grid item xs={12} lg={6}>
                            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                                <Typography variant="h6" gutterBottom>
                                    Data Timeline
                                </Typography>
                                <Box sx={{ flexGrow: 1, minHeight: 300 }}>
                                    <div ref={dygraphRef} style={{ width: '100%', height: '100%' }}></div>
                                </Box>
                                {episodeData.ignored_columns && episodeData.ignored_columns.length > 0 && (
                                    <Alert severity="info" sx={{ mt: 1 }}>
                                        Ignored columns (due to &gt;1D data): {episodeData.ignored_columns.join(', ')}
                                    </Alert>
                                )}
                            </Paper>
                        </Grid>
                    </Grid>
                </Box>

                {/* Controls Bar */}
                <Paper elevation={3} sx={{ p: 2, bgcolor: 'background.paper' }}>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item>
                            <IconButton onClick={handlePlayPause} color="primary" size="large">
                                {isPlaying ? <PauseIcon /> : <PlayIcon />}
                            </IconButton>
                        </Grid>
                        
                        <Grid item>
                            <IconButton onClick={() => jumpToTime(-5)} size="small">
                                <PrevIcon />
                            </IconButton>
                        </Grid>
                        
                        <Grid item>
                            <IconButton onClick={() => jumpToTime(5)} size="small">
                                <NextIcon />
                            </IconButton>
                        </Grid>
                        
                        <Grid item>
                            <IconButton 
                                onClick={() => {
                                    Object.values(videoRefs.current).forEach(video => {
                                        if (video) video.currentTime = 0;
                                    });
                                }} 
                                size="small"
                            >
                                <ReplayIcon />
                            </IconButton>
                        </Grid>
                        
                        <Grid item sx={{ flexGrow: 1, px: 2 }}>
                            <Slider
                                value={sliderValue}
                                onChange={handleSliderChange}
                                onChangeCommitted={handleSliderChangeCommitted}
                                min={0}
                                max={100}
                                step={0.1}
                                valueLabelDisplay="auto"
                                valueLabelFormat={(value) => formatTime((value / 100) * mainVideoDuration)}
                            />
                        </Grid>
                        
                        <Grid item>
                            <Typography variant="body2" sx={{ minWidth: 120, textAlign: 'center' }}>
                                {formatTime(currentVideoTime)} / {formatTime(mainVideoDuration)}
                            </Typography>
                        </Grid>
                    </Grid>
                </Paper>
            </Box>
        </Box>
    );
}

export default EpisodeViewPage; 