// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Class colors
const CLASS_COLORS = {
    'pothole': '#FF6347',
    'longitudinal_crack': '#FFD700',
    'crazing': '#32CD32',
    'faded_marking': '#6495ED'
};

const CLASS_NAMES = {
    'pothole': 'Pothole',
    'longitudinal_crack': 'Longitudinal Crack',
    'crazing': 'Crazing',
    'faded_marking': 'Faded Marking'
};

// Global variables
let map;
let markersLayer;
let detections = [];
let filteredDetections = [];

// Initialize map
function initMap() {
    // Create map centered on Paris (default)
    map = L.map('map').setView([48.8566, 2.3522], 13);
    
    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Create marker cluster group
    markersLayer = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true
    });
    
    map.addLayer(markersLayer);
    
    console.log('Map initialized');
}

// Update statistics
function updateStatistics() {
    const stats = {
        total: 0,
        pothole: 0,
        longitudinal_crack: 0,
        crazing: 0,
        faded_marking: 0
    };
    
    filteredDetections.forEach(det => {
        stats.total++;
        stats[det.class_name]++;
    });
    
    document.getElementById('totalDetections').textContent = stats.total;
    document.getElementById('totalPotholes').textContent = stats.pothole;
    document.getElementById('totalCracks').textContent = stats.longitudinal_crack;
    document.getElementById('totalCrazing').textContent = stats.crazing;
    document.getElementById('totalMarkings').textContent = stats.faded_marking;
}

// Create marker icon
function createMarkerIcon(className) {
    return L.divIcon({
        className: 'custom-marker',
        html: `<div style="background-color: ${CLASS_COLORS[className]}; width: 30px; height: 30px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"></div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });
}

// Add markers to map
function addMarkersToMap(data) {
    markersLayer.clearLayers();
    
    data.forEach(det => {
        if (det.latitude && det.longitude) {
            const marker = L.marker(
                [det.latitude, det.longitude],
                { icon: createMarkerIcon(det.class_name) }
            );
            
            // Create popup content
            const popupContent = `
                <div class="popup-content">
                    <h3>${CLASS_NAMES[det.class_name]}</h3>
                    <p><strong>Confidence:</strong> ${(det.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Location:</strong> ${det.latitude.toFixed(6)}, ${det.longitude.toFixed(6)}</p>
                    ${det.timestamp ? `<p><strong>Time:</strong> ${new Date(det.timestamp).toLocaleString()}</p>` : ''}
                    ${det.frame_number !== undefined ? `<p><strong>Frame:</strong> ${det.frame_number}</p>` : ''}
                </div>
            `;
            
            marker.bindPopup(popupContent);
            markersLayer.addLayer(marker);
        }
    });
    
    // Fit bounds if we have markers
    if (data.length > 0 && data.some(d => d.latitude && d.longitude)) {
        const bounds = markersLayer.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [50, 50] });
        }
    }
}

// Filter detections
function filterDetections() {
    const activeClasses = Array.from(document.querySelectorAll('.filter-class:checked'))
        .map(cb => cb.value);
    
    filteredDetections = detections.filter(det => activeClasses.includes(det.class_name));
    
    addMarkersToMap(filteredDetections);
    updateStatistics();
}

// Load GeoJSON file
async function loadGeoJSON(file) {
    try {
        const text = await file.text();
        const geojson = JSON.parse(text);
        
        if (geojson.type === 'FeatureCollection') {
            detections = geojson.features.map(feature => ({
                latitude: feature.geometry.coordinates[1],
                longitude: feature.geometry.coordinates[0],
                class_name: feature.properties.class,
                class_id: feature.properties.class_id,
                confidence: feature.properties.confidence,
                timestamp: feature.properties.timestamp,
                frame_number: feature.properties.frame_number
            }));
            
            filterDetections();
            showNotification('GeoJSON loaded successfully!', 'success');
        }
    } catch (error) {
        console.error('Error loading GeoJSON:', error);
        showNotification('Error loading GeoJSON file', 'error');
    }
}

// Upload image
async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', document.getElementById('confThreshold').value);
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/detect/image`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Add detections
            detections = result.detections.map(det => ({
                ...det,
                latitude: det.latitude || 48.8566 + Math.random() * 0.01,
                longitude: det.longitude || 2.3522 + Math.random() * 0.01
            }));
            
            filterDetections();
            showNotification(`Found ${result.num_detections} detections!`, 'success');
            
            // Show annotated image
            if (result.annotated_image) {
                showResultsModal(result);
            }
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        showNotification('Error processing image', 'error');
    } finally {
        showLoading(false);
    }
}

// Upload video
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', document.getElementById('confThreshold').value);
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/detect/video`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Video processing started...', 'info');
            pollJobStatus(result.job_id);
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        showNotification('Error uploading video', 'error');
        showLoading(false);
    }
}

// Poll job status
async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/job/${jobId}`);
            const job = await response.json();
            
            if (job.status === 'completed') {
                clearInterval(interval);
                showLoading(false);
                
                // Load results
                const resultResponse = await fetch(`${API_BASE_URL}${job.result_path}`);
                const resultData = await resultResponse.json();
                
                detections = resultData.detections;
                filterDetections();
                
                showNotification(`Processing complete! Found ${job.num_detections} detections`, 'success');
            } else if (job.status === 'failed') {
                clearInterval(interval);
                showLoading(false);
                showNotification('Processing failed', 'error');
            }
        } catch (error) {
            console.error('Error polling job:', error);
            clearInterval(interval);
            showLoading(false);
        }
    }, 2000);
}

// Show/hide loading overlay
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// Show notification
function showNotification(message, type = 'info') {
    // Simple alert for now - could be enhanced with a toast library
    alert(message);
}

// Show results modal
function showResultsModal(result) {
    const modal = document.getElementById('resultsModal');
    const content = document.getElementById('resultsContent');
    
    content.innerHTML = `
        <img src="${API_BASE_URL}${result.annotated_image}" style="max-width: 100%; border-radius: 8px;" />
        <p style="margin-top: 15px;"><strong>Detections:</strong> ${result.num_detections}</p>
    `;
    
    modal.style.display = 'flex';
}

// Export to GeoJSON
function exportGeoJSON() {
    const geojson = {
        type: 'FeatureCollection',
        features: filteredDetections.map(det => ({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [det.longitude, det.latitude]
            },
            properties: {
                class: det.class_name,
                class_id: det.class_id,
                confidence: det.confidence,
                timestamp: det.timestamp,
                frame_number: det.frame_number
            }
        }))
    };
    
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detections_${new Date().getTime()}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
}

// Export to CSV
function exportCSV() {
    const headers = ['Class', 'Confidence', 'Latitude', 'Longitude', 'Timestamp', 'Frame'];
    const rows = filteredDetections.map(det => [
        det.class_name,
        det.confidence,
        det.latitude,
        det.longitude,
        det.timestamp || '',
        det.frame_number || ''
    ]);
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detections_${new Date().getTime()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    
    // Confidence threshold slider
    const confSlider = document.getElementById('confThreshold');
    const confValue = document.getElementById('confValue');
    confSlider.addEventListener('input', (e) => {
        confValue.textContent = e.target.value;
    });
    
    // Image upload
    document.getElementById('imageUpload').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadImage(file);
        }
    });
    
    // Video upload
    document.getElementById('videoUpload').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadVideo(file);
        }
    });
    
    // Class filters
    document.querySelectorAll('.filter-class').forEach(checkbox => {
        checkbox.addEventListener('change', filterDetections);
    });
    
    // Export buttons
    document.getElementById('exportGeoJSON').addEventListener('click', exportGeoJSON);
    document.getElementById('exportCSV').addEventListener('click', exportCSV);
    
    // Modal close
    document.querySelector('.close').addEventListener('click', () => {
        document.getElementById('resultsModal').style.display = 'none';
    });
    
    window.addEventListener('click', (e) => {
        const modal = document.getElementById('resultsModal');
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Load sample data if available
    loadSampleData();
});

// Load sample data (for demo purposes)
async function loadSampleData() {
    // This would load from a sample GeoJSON file if available
    // For now, we'll create some dummy data
    const sampleDetections = [
        {
            latitude: 48.8566,
            longitude: 2.3522,
            class_name: 'pothole',
            class_id: 0,
            confidence: 0.95,
            timestamp: new Date().toISOString()
        },
        {
            latitude: 48.8576,
            longitude: 2.3532,
            class_name: 'longitudinal_crack',
            class_id: 1,
            confidence: 0.87,
            timestamp: new Date().toISOString()
        },
        {
            latitude: 48.8586,
            longitude: 2.3542,
            class_name: 'crazing',
            class_id: 2,
            confidence: 0.92,
            timestamp: new Date().toISOString()
        }
    ];
    
    // Uncomment to show sample data
    // detections = sampleDetections;
    // filterDetections();
}
