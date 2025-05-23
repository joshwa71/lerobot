# LeRobot Dataset Visualizer

A modern web application for visualizing LeRobot datasets with a Flask backend and React frontend.

## Features

- **Modern UI**: Built with Material-UI for a professional, responsive interface
- **Dataset Browser**: Grid-based view of available datasets with key metrics
- **Episode Visualization**: 
  - Side-by-side video streams and data charts
  - Interactive timeline with Dygraphs
  - Synchronized video playback and data exploration
  - Professional controls for navigation and playback
- **Dataset Merging**: 
  - Select any two datasets to merge into a new combined dataset
  - Real-time progress tracking and status updates
  - Automatic refresh of dataset list after successful merge

## Architecture

- **Backend**: Flask API serving dataset data and video files
- **Frontend**: React application with Material-UI components
- **Data Visualization**: Dygraphs for interactive time-series data
- **Video Playback**: HTML5 video with custom controls

## Setup

### Prerequisites

- Python 3.8+ with LeRobot installed
- Node.js 16+ and npm
- LeRobot datasets in your outputs directory

### Installation

1. **Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Frontend Dependencies**:
   ```bash
   cd frontend/dataset_visualiser
   npm install
   ```

## Running the Application

### Start Backend
```bash
cd backend
python app.py --lerobot_root_path /path/to/your/lerobot/outputs
```

The backend will run on `http://localhost:5001` by default.

### Start Frontend
```bash
cd frontend/dataset_visualiser
npm start
```

The frontend will run on `http://localhost:3000` by default.

### Custom Configuration

Backend options:
- `--host`: Backend host (default: 127.0.0.1)
- `--port`: Backend port (default: 5001)
- `--lerobot_root_path`: Path to LeRobot datasets directory (required)

Frontend configuration:
- Set `REACT_APP_API_URL` environment variable to point to backend if needed

## Usage

1. **Browse Datasets**: The home page shows all available datasets in your configured directory
2. **Select Dataset**: Click on any dataset card to start exploring its episodes
3. **Merge Datasets**: 
   - Select checkboxes on any two datasets you want to merge
   - Click "Merge Selected Datasets" button
   - Enter a name for the new merged dataset
   - Confirm the merge and monitor progress in real-time
4. **Navigate Episodes**: Use the sidebar controls to navigate between episodes
5. **Control Playbook**: Use the bottom control bar to play/pause videos and scrub through time
6. **Analyze Data**: The interactive chart shows sensor data synchronized with video playback
7. **Filter Videos**: Use the sidebar checkboxes to show/hide specific video streams

## API Endpoints

### Dataset Operations
- `GET /api/datasets` - List all available datasets
- `GET /api/datasets/<repo_id>/metadata` - Get dataset metadata
- `GET /api/datasets/<repo_id>/episodes/<episode_id>/data` - Get episode data and videos
- `GET /api/datasets/<repo_id>/videos/<video_path>` - Serve video files

### Dataset Merging
- `POST /api/datasets/merge` - Start a dataset merge operation
  - Body: `{"source1": "dataset1", "source2": "dataset2", "target_name": "merged_dataset"}`
  - Returns: `{"operation_id": "uuid", "message": "Merge operation started"}`
- `GET /api/datasets/merge/<operation_id>/status` - Check merge operation status
  - Returns: `{"status": "running|completed|error", "message": "...", ...}`

## File Structure

```
dataset_app/
├── backend/
│   ├── app.py              # Flask application with merge functionality
│   └── requirements.txt    # Python dependencies
└── frontend/
    └── dataset_visualiser/
        ├── src/
        │   ├── components/
        │   │   ├── DatasetListPage.js   # Dataset browser with merge controls
        │   │   └── EpisodeViewPage.js   # Episode visualization
        │   └── App.js                   # Main app with routing
        └── package.json                 # Node dependencies
```

## Key Improvements Over Original

- **Professional Layout**: Modern dashboard-style interface with proper sections
- **Better Navigation**: Dedicated sidebar with episode selection and video controls  
- **Responsive Design**: Works well on different screen sizes
- **Material Design**: Consistent, professional styling throughout
- **Enhanced Controls**: Better video playback controls with proper synchronization
- **Performance**: Optimized event handling to prevent cursor/interaction issues
- **Dataset Merging**: Complete workflow for combining datasets with progress tracking

## Dataset Merging Details

The dataset merging functionality allows you to combine two existing LeRobot datasets into a new merged dataset. This is useful for:

- Combining data collected in separate sessions
- Creating larger training datasets from multiple smaller ones
- Consolidating datasets with similar tasks or robot configurations

### Merge Process

1. **Selection**: Use checkboxes to select exactly 2 datasets
2. **Configuration**: Enter a name for the new merged dataset
3. **Execution**: The merge operation runs asynchronously in the background
4. **Progress**: Real-time status updates show merge progress
5. **Completion**: The new dataset appears in your dataset list automatically

### Technical Details

- Merges episode indices and frame indices to avoid conflicts
- Handles task mapping and deduplication
- Copies all data files (.parquet) and video files (.mp4)
- Updates metadata to reflect the combined dataset
- Supports single-chunk datasets (most common case)
- Creates proper LeRobot dataset structure in the target location

## Troubleshooting

- **Datasets not showing**: Ensure `lerobot_root_path` points to correct directory
- **Videos not playing**: Check browser codec support for your video files
- **Backend errors**: Check that LeRobot is properly installed and datasets are accessible
- **CORS issues**: Ensure backend and frontend are running on expected ports
- **Merge failures**: Check backend logs for detailed error messages
- **Merge conflicts**: Ensure target dataset name doesn't already exist 