/**
 * Flight Delay Predictor - Frontend JavaScript
 * Handles map visualization, form controls, and API interactions
 */

// =============================================================================
// GLOBAL STATE
// =============================================================================
let airports = [];
let airlines = [];
let mapState = {
    svg: null,
    g: null,
    projection: null,
    selectedOrigin: null,
    selectedDest: null
};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    // Set current year in footer
    document.getElementById('year').textContent = new Date().getFullYear();
    
    // Initialize mobile menu
    initMobileMenu();
    
    // Load data from API
    await loadAirports();
    await loadAirlines();
    await loadModelInfo();
    
    // Initialize components
    initializeMap();
    initializeControls();
});

// =============================================================================
// DATA LOADING
// =============================================================================
async function loadAirports() {
    try {
        const response = await fetch('/api/airports');
        airports = await response.json();
        populateAirportDropdowns();
    } catch (error) {
        console.error('Failed to load airports:', error);
        // Fallback to default airports
        airports = getDefaultAirports();
        populateAirportDropdowns();
    }
}

async function loadAirlines() {
    try {
        const response = await fetch('/api/airlines');
        airlines = await response.json();
        populateAirlineDropdown();
    } catch (error) {
        console.error('Failed to load airlines:', error);
        // Fallback to default airlines
        airlines = getDefaultAirlines();
        populateAirlineDropdown();
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const info = await response.json();
        
        if (info.metrics) {
            document.getElementById('metric-r2').textContent = info.metrics.r2Score.toFixed(3);
            document.getElementById('metric-rmse').textContent = info.metrics.rmse.toFixed(1);
            document.getElementById('metric-mae').textContent = info.metrics.mae.toFixed(1);
            document.getElementById('metric-mape').textContent = info.metrics.mape.toFixed(1) + '%';
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

// =============================================================================
// DEFAULT DATA (Fallback)
// =============================================================================
function getDefaultAirports() {
    return [
        {code: 'ATL', city: 'Atlanta', state: 'GA', lat: 33.6367, lon: -84.4281},
        {code: 'ORD', city: 'Chicago', state: 'IL', lat: 41.9786, lon: -87.9048},
        {code: 'DFW', city: 'Dallas/Fort Worth', state: 'TX', lat: 32.8968, lon: -97.038},
        {code: 'DEN', city: 'Denver', state: 'CO', lat: 39.8617, lon: -104.673},
        {code: 'LAX', city: 'Los Angeles', state: 'CA', lat: 33.9425, lon: -118.408},
        {code: 'JFK', city: 'New York JFK', state: 'NY', lat: 40.6394, lon: -73.7793},
        {code: 'SFO', city: 'San Francisco', state: 'CA', lat: 37.6198, lon: -122.3748},
        {code: 'SEA', city: 'Seattle', state: 'WA', lat: 47.4479, lon: -122.3103},
        {code: 'MIA', city: 'Miami', state: 'FL', lat: 25.7932, lon: -80.2906},
        {code: 'BOS', city: 'Boston', state: 'MA', lat: 42.362, lon: -71.0079},
        {code: 'PHX', city: 'Phoenix', state: 'AZ', lat: 33.4353, lon: -112.0059},
        {code: 'IAH', city: 'Houston', state: 'TX', lat: 29.9844, lon: -95.3414},
        {code: 'LAS', city: 'Las Vegas', state: 'NV', lat: 36.0834, lon: -115.1518},
        {code: 'MCO', city: 'Orlando', state: 'FL', lat: 28.4294, lon: -81.309},
        {code: 'CLT', city: 'Charlotte', state: 'NC', lat: 35.214, lon: -80.9431},
        {code: 'EWR', city: 'Newark', state: 'NJ', lat: 40.6925, lon: -74.1687},
        {code: 'MSP', city: 'Minneapolis', state: 'MN', lat: 44.8801, lon: -93.2217},
        {code: 'DTW', city: 'Detroit', state: 'MI', lat: 42.2138, lon: -83.3538},
        {code: 'PHL', city: 'Philadelphia', state: 'PA', lat: 39.8719, lon: -75.2411},
        {code: 'SLC', city: 'Salt Lake City', state: 'UT', lat: 40.7889, lon: -111.9799},
        {code: 'LGA', city: 'New York LaGuardia', state: 'NY', lat: 40.7772, lon: -73.8726},
        {code: 'BWI', city: 'Baltimore', state: 'MD', lat: 39.1754, lon: -76.6683},
        {code: 'DCA', city: 'Washington Reagan', state: 'DC', lat: 38.8521, lon: -77.0377},
        {code: 'SAN', city: 'San Diego', state: 'CA', lat: 32.7336, lon: -117.19},
        {code: 'TPA', city: 'Tampa', state: 'FL', lat: 27.9755, lon: -82.5332}
    ];
}

function getDefaultAirlines() {
    return [
        {code: 'AA', name: 'American Airlines'},
        {code: 'DL', name: 'Delta Air Lines'},
        {code: 'UA', name: 'United Airlines'},
        {code: 'WN', name: 'Southwest Airlines'},
        {code: 'B6', name: 'JetBlue Airways'},
        {code: 'AS', name: 'Alaska Airlines'},
        {code: 'NK', name: 'Spirit Airlines'},
        {code: 'F9', name: 'Frontier Airlines'}
    ];
}

// =============================================================================
// DROPDOWN POPULATION
// =============================================================================
function populateAirportDropdowns() {
    const originSelect = document.getElementById('origin-select');
    const destSelect = document.getElementById('dest-select');
    
    airports.forEach(airport => {
        const text = `${airport.code} - ${airport.city}, ${airport.state}`;
        
        const originOption = document.createElement('option');
        originOption.value = airport.code;
        originOption.textContent = text;
        originSelect.appendChild(originOption);
        
        const destOption = document.createElement('option');
        destOption.value = airport.code;
        destOption.textContent = text;
        destSelect.appendChild(destOption);
    });
    
    originSelect.addEventListener('change', handleOriginDropdown);
    destSelect.addEventListener('change', handleDestDropdown);
}

function populateAirlineDropdown() {
    const airlineSelect = document.getElementById('airline-select');
    
    airlines.forEach(airline => {
        const option = document.createElement('option');
        option.value = airline.code;
        option.textContent = `${airline.name} (${airline.code})`;
        airlineSelect.appendChild(option);
    });
    
    airlineSelect.addEventListener('change', checkFormComplete);
}

// =============================================================================
// MAP INITIALIZATION
// =============================================================================
async function initializeMap() {
    const container = document.getElementById('us-map');
    const width = container.clientWidth;
    const height = 380;
    
    mapState.svg = d3.select('#us-map')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    mapState.g = mapState.svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([1, 6])
        .on('zoom', (e) => {
            mapState.g.attr('transform', e.transform);
            const s = e.transform.k;
            mapState.g.selectAll('.airport circle')
                .attr('r', function() {
                    const isSelected = d3.select(this.parentNode).classed('selected-origin') ||
                                       d3.select(this.parentNode).classed('selected-dest');
                    return Math.max(isSelected ? 8/s : 5/s, 2);
                });
        });
    
    mapState.svg.call(zoom);
    
    // Set up projection
    mapState.projection = d3.geoAlbersUsa()
        .scale(width * 1.3)
        .translate([width / 2, height / 2]);
    
    const path = d3.geoPath().projection(mapState.projection);
    
    // Load US map
    try {
        const usData = await d3.json('https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json');
        mapState.g.append('g')
            .selectAll('path')
            .data(topojson.feature(usData, usData.objects.states).features)
            .enter()
            .append('path')
            .attr('class', 'state')
            .attr('d', path);
    } catch (error) {
        console.error('Failed to load US map:', error);
    }
    
    // Render airports
    renderAirports();
}

function renderAirports() {
    const airportsGroup = mapState.g.append('g').attr('class', 'airports');
    
    airportsGroup.selectAll('.airport')
        .data(airports)
        .enter()
        .append('g')
        .attr('class', 'airport')
        .attr('transform', d => {
            const coords = mapState.projection([d.lon, d.lat]);
            return coords ? `translate(${coords[0]}, ${coords[1]})` : null;
        })
        .on('click', handleAirportClick)
        .on('mouseenter', handleAirportHover)
        .on('mouseleave', () => {
            document.getElementById('map-tooltip').style.display = 'none';
        })
        .append('circle')
        .attr('r', 5);
}

// =============================================================================
// MAP INTERACTIONS
// =============================================================================
function handleAirportClick(event, airport) {
    if (!mapState.selectedOrigin) {
        // Set origin
        mapState.selectedOrigin = airport;
        document.getElementById('origin-select').value = airport.code;
    } else if (!mapState.selectedDest && airport.code !== mapState.selectedOrigin.code) {
        // Set destination
        mapState.selectedDest = airport;
        document.getElementById('dest-select').value = airport.code;
        drawRouteLine();
    } else {
        // Reset and set new origin
        clearSelection();
        mapState.selectedOrigin = airport;
        document.getElementById('origin-select').value = airport.code;
    }
    
    updateMapSelection();
    updateRouteInfo();
    checkFormComplete();
}

function handleOriginDropdown() {
    const code = document.getElementById('origin-select').value;
    if (code) {
        mapState.selectedOrigin = airports.find(a => a.code === code);
        updateMapSelection();
        updateRouteInfo();
        if (mapState.selectedDest) drawRouteLine();
        checkFormComplete();
    }
}

function handleDestDropdown() {
    const code = document.getElementById('dest-select').value;
    if (code && (!mapState.selectedOrigin || code !== mapState.selectedOrigin.code)) {
        mapState.selectedDest = airports.find(a => a.code === code);
        updateMapSelection();
        updateRouteInfo();
        drawRouteLine();
        checkFormComplete();
    }
}

function handleAirportHover(event, airport) {
    const tooltip = document.getElementById('map-tooltip');
    tooltip.innerHTML = `<strong>${airport.code}</strong><br>${airport.city}, ${airport.state}`;
    tooltip.style.left = (event.offsetX + 15) + 'px';
    tooltip.style.top = (event.offsetY - 10) + 'px';
    tooltip.style.display = 'block';
}

function updateMapSelection() {
    mapState.g.selectAll('.airport')
        .classed('selected-origin', d => mapState.selectedOrigin && d.code === mapState.selectedOrigin.code)
        .classed('selected-dest', d => mapState.selectedDest && d.code === mapState.selectedDest.code);
    
    mapState.g.selectAll('.airport circle')
        .attr('r', d => {
            const isSelected = (mapState.selectedOrigin && d.code === mapState.selectedOrigin.code) ||
                               (mapState.selectedDest && d.code === mapState.selectedDest.code);
            return isSelected ? 8 : 5;
        });
}

function updateRouteInfo() {
    const routeInfo = document.getElementById('route-info');
    
    if (mapState.selectedOrigin && mapState.selectedDest) {
        const distance = calculateDistance(
            mapState.selectedOrigin.lat, mapState.selectedOrigin.lon,
            mapState.selectedDest.lat, mapState.selectedDest.lon
        );
        const duration = Math.round(distance / 500 * 60 + 30);
        
        document.getElementById('route-distance').textContent = `${Math.round(distance)} mi`;
        document.getElementById('route-duration').textContent = `~${Math.floor(duration/60)}h ${duration%60}m`;
        routeInfo.classList.add('visible');
    } else {
        routeInfo.classList.remove('visible');
    }
}

function drawRouteLine() {
    mapState.g.selectAll('.route-line').remove();
    
    if (mapState.selectedOrigin && mapState.selectedDest) {
        const origin = mapState.projection([mapState.selectedOrigin.lon, mapState.selectedOrigin.lat]);
        const dest = mapState.projection([mapState.selectedDest.lon, mapState.selectedDest.lat]);
        
        if (origin && dest) {
            const mx = (origin[0] + dest[0]) / 2;
            const my = (origin[1] + dest[1]) / 2 - 50;
            
            mapState.g.append('path')
                .attr('class', 'route-line')
                .attr('d', `M${origin[0]},${origin[1]} Q${mx},${my} ${dest[0]},${dest[1]}`);
        }
    }
}

function clearSelection() {
    mapState.selectedOrigin = null;
    mapState.selectedDest = null;
    mapState.g.selectAll('.route-line').remove();
    document.getElementById('origin-select').value = '';
    document.getElementById('dest-select').value = '';
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 3959; // Earth's radius in miles
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) ** 2 + 
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
              Math.sin(dLon/2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// =============================================================================
// FORM CONTROLS
// =============================================================================
function initializeControls() {
    const depSlider = document.getElementById('dep-hour-slider');
    const arrSlider = document.getElementById('arr-hour-slider');
    
    depSlider.addEventListener('input', () => {
        document.getElementById('dep-time-display').textContent = formatHour(depSlider.value);
        updateSliderGradient(depSlider);
    });
    
    arrSlider.addEventListener('input', () => {
        document.getElementById('arr-time-display').textContent = formatHour(arrSlider.value);
        updateSliderGradient(arrSlider);
    });
    
    updateSliderGradient(depSlider);
    updateSliderGradient(arrSlider);
    
    document.getElementById('predict-button').addEventListener('click', makePrediction);
}

function formatHour(h) {
    return `${h.toString().padStart(2, '0')}:00`;
}

function updateSliderGradient(slider) {
    const percent = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #ca9a4e 0%, #ca9a4e ${percent}%, #230505 ${percent}%, #230505 100%)`;
}

function checkFormComplete() {
    const originSelected = mapState.selectedOrigin !== null;
    const destSelected = mapState.selectedDest !== null;
    const airlineSelected = document.getElementById('airline-select').value !== '';
    
    document.getElementById('predict-button').disabled = !(originSelected && destSelected && airlineSelected);
}

function initMobileMenu() {
    document.getElementById('mobile-toggle').addEventListener('click', () => {
        document.getElementById('nav-menu').classList.toggle('active');
    });
}

// =============================================================================
// PREDICTION
// =============================================================================
async function makePrediction() {
    // Show loading overlay
    document.getElementById('loading-overlay').classList.add('visible');
    
    // Gather form data
    const data = {
        origin: mapState.selectedOrigin.code,
        destination: mapState.selectedDest.code,
        month: document.getElementById('month-select').value,
        day: document.getElementById('day-input').value,
        dayOfWeek: document.getElementById('day-of-week-select').value,
        depHour: document.getElementById('dep-hour-slider').value,
        arrHour: document.getElementById('arr-hour-slider').value,
        airline: document.getElementById('airline-select').value
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            alert('Prediction failed: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Failed to make prediction. Please try again.');
    } finally {
        document.getElementById('loading-overlay').classList.remove('visible');
    }
}

// =============================================================================
// RESULTS DISPLAY
// =============================================================================
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.add('visible');
    
    // Probability
    const probPercent = result.probabilityPercent;
    document.getElementById('prob-value').textContent = probPercent + '%';
    setTimeout(() => {
        document.getElementById('prob-bar').style.width = probPercent + '%';
    }, 100);
    
    // Risk styling
    const probValue = document.getElementById('prob-value');
    probValue.className = 'result-value risk-' + result.riskLevel;
    document.getElementById('prob-description').textContent = result.riskText;
    
    // Duration
    const delay = result.expectedDelay;
    if (delay > 0) {
        const hours = Math.floor(delay / 60);
        const mins = Math.round(delay % 60);
        document.getElementById('duration-value').textContent = hours > 0 ? `${hours}h ${mins}m` : `${Math.round(delay)} min`;
        document.getElementById('duration-value').className = 'result-value risk-' + (delay > 60 ? 'high' : delay > 30 ? 'medium' : 'low');
        setTimeout(() => {
            document.getElementById('duration-bar').style.width = Math.min(delay / 180 * 100, 100) + '%';
        }, 100);
        document.getElementById('duration-description').textContent = 'Expected delay if flight is delayed';
    } else {
        document.getElementById('duration-value').textContent = 'On Time';
        document.getElementById('duration-value').className = 'result-value risk-low';
        document.getElementById('duration-bar').style.width = '0%';
        document.getElementById('duration-description').textContent = 'Flight likely to arrive on schedule';
    }
    
    // SHAP values
    displayShapChart(result.shapValues);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayShapChart(shapValues) {
    const chart = document.getElementById('shap-chart');
    chart.innerHTML = '';
    
    if (!shapValues || shapValues.length === 0) return;
    
    const maxAbs = Math.max(...shapValues.map(d => Math.abs(d.shap)));
    
    shapValues.forEach((item, i) => {
        const container = document.createElement('div');
        container.className = 'shap-bar-container';
        
        const name = document.createElement('div');
        name.className = 'shap-feature-name';
        name.innerHTML = `${item.displayName}: <span class="shap-feature-value">${item.value}</span>`;
        
        const track = document.createElement('div');
        track.className = 'shap-bar-track';
        
        const bar = document.createElement('div');
        bar.className = 'shap-bar ' + (item.shap > 0 ? 'shap-bar-positive' : 'shap-bar-negative');
        bar.style.width = '0%';
        
        const percent = (Math.abs(item.shap) / maxAbs) * 100;
        setTimeout(() => {
            bar.style.width = percent + '%';
        }, 100 + i * 50);
        
        track.appendChild(bar);
        container.appendChild(name);
        container.appendChild(track);
        chart.appendChild(container);
    });
}
