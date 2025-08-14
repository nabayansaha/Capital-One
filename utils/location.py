import math
import requests
from typing import Dict, Tuple

# Store last GPS location received from phone/browser
last_gps_location = None

def set_gps_location(lat: float, lon: float, accuracy: float = None):
    """Update last GPS location (called by Flask route)."""
    global last_gps_location
    last_gps_location = {
        'latitude': lat,
        'longitude': lon,
        'accuracy': accuracy
    }

def get_user_location() -> Dict:
    """Get user location from GPS if available, otherwise fall back to IP-based location."""
    global last_gps_location
    if last_gps_location:
        return {
            'source': 'gps',
            **last_gps_location
        }
    
    # Fallback: IP-based geolocation
    try:
        response = requests.get('http://ip-api.com/json/')
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success':
            return {
                'source': 'ip',
                'latitude': data['lat'],
                'longitude': data['lon'],
                'city': data['city'],
                'region': data['regionName'],
                'country': data['country'],
                'timezone': data['timezone']
            }
        else:
            return {'error': 'Failed to get location'}
    except requests.RequestException as e:
        return {'error': f'Request failed: {str(e)}'}

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile
