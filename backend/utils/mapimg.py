import requests
import os
from dotenv import load_dotenv
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import io
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from location import get_user_location, deg2num

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('satellite_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_google_satellite_tile_urls(lat: float, lon: float, zoom: int = 18, tile_size: int = 3) -> Dict:
    """Generate Google Satellite tile URLs"""
    
    center_x, center_y = deg2num(lat, lon, zoom)

    source_info = {
        'url_template': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'max_zoom': 20,
        'description': 'Google Satellite - Highest resolution available'
    }
    
    tiles = []
    for dx in range(-tile_size//2, tile_size//2 + 1):
        for dy in range(-tile_size//2, tile_size//2 + 1):
            x = center_x + dx
            y = center_y + dy
            
            url = source_info['url_template'].format(z=zoom, x=x, y=y)
            
            tiles.append({
                'url': url,
                'x': x,
                'y': y,
                'pos': (dx, dy)
            })
    
    tile_urls = {
        'google_satellite': {
            'tiles': tiles,
            'description': source_info['description']
        }
    }
    
    return tile_urls

def download_tile(tile_info: Dict, source_name: str) -> Optional[Image.Image]:
    """Download a single tile with appropriate headers"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://maps.google.com/'
        }
        
        response = requests.get(tile_info['url'], headers=headers, timeout=15)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        return image
        
    except Exception as e:
        logger.error(f"Failed to download tile from {source_name}: {str(e)}")
        return None

def stitch_tiles(tiles: list, tile_size: int) -> Image.Image:
    """Stitch multiple tiles into a single high-resolution image"""
    if not tiles:
        return None
    
    # Standard tile size is 256x256
    tile_width = tile_height = 256
    
    # Create a large image to hold all tiles
    total_width = tile_size * tile_width
    total_height = tile_size * tile_height
    
    stitched = Image.new('RGB', (total_width, total_height))
    
    valid_tiles = 0
    for tile_info in tiles:
        if tile_info['image'] is not None:
            dx, dy = tile_info['pos']
            x_offset = (dx + tile_size//2) * tile_width
            y_offset = (dy + tile_size//2) * tile_height
            
            stitched.paste(tile_info['image'], (x_offset, y_offset))
            valid_tiles += 1
    
    if valid_tiles == 0:
        return None
    
    return stitched

def get_google_satellite_imagery(lat: float, lon: float, zoom: int = 19, tile_grid: int = 5) -> Optional[Image.Image]:
    """Get high resolution satellite imagery using Google Satellite only"""
    
    logger.info(f"Downloading {tile_grid}x{tile_grid} = {tile_grid**2} tiles at zoom level {zoom}")
    logger.info(f"Target resolution: ~{256 * tile_grid}x{256 * tile_grid} pixels")
    
    tile_data = get_google_satellite_tile_urls(lat, lon, zoom, tile_grid)
    
    source_name = 'google_satellite'
    source_info = tile_data[source_name]
    
    logger.info(f"Processing {source_name} ({source_info['description']})")
    tiles = source_info['tiles']
    
    downloaded_tiles = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, tile_info in enumerate(tiles):
            time.sleep(0.1)
            future = executor.submit(download_tile, tile_info, source_name)
            futures.append((future, tile_info))
        
        # Collect results
        successful_downloads = 0
        for future, tile_info in futures:
            try:
                image = future.result(timeout=20)
                tile_info['image'] = image
                if image:
                    successful_downloads += 1
            except Exception as e:
                tile_info['image'] = None
            downloaded_tiles.append(tile_info)
    
    logger.info(f"Downloaded {successful_downloads}/{len(tiles)} tiles")
    
    # Stitch tiles together
    if successful_downloads > 0:
        stitched_image = stitch_tiles(downloaded_tiles, tile_grid)
        
        if stitched_image:
            logger.info(f"Final image: {stitched_image.size[0]}x{stitched_image.size[1]} pixels")
            return stitched_image
        else:
            logger.error("Failed to stitch tiles")
    else:
        logger.error("No tiles downloaded successfully")
    
    return None

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Apply basic image enhancement to improve quality"""
    try:
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply slight contrast enhancement
        enhanced = np.clip(img_array * 1.1, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced)
        
        return enhanced_image
    except:
        # Return original image if enhancement fails
        return image

def display_satellite_image(image: Image.Image, lat: float, lon: float):
    """Display the satellite image"""
    
    if not image:
        logger.error("No image to display")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Apply enhancement
    enhanced_image = enhance_image_quality(image)
    
    ax.imshow(enhanced_image)
    ax.set_title(f"Google Satellite Imagery\n{image.size[0]}x{image.size[1]} pixels\nLat: {lat:.6f}, Lon: {lon:.6f}", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_satellite_image(image: Image.Image, lat: float, lon: float, base_filename: str = None):
    """Save satellite image to disk"""
    
    if base_filename is None:
        base_filename = f"google_satellite_highres_{lat:.4f}_{lon:.4f}"
    
    # Apply enhancement before saving
    enhanced_image = enhance_image_quality(image)
    
    filename = f"{base_filename}.png"
    
    try:
        # Save at maximum quality
        enhanced_image.save(filename, 'PNG', optimize=True)
        file_size = os.path.getsize(filename) / 1024 / 1024  # MB
        
        logger.info(f"Saved {filename}")
        logger.info(f"Size: {enhanced_image.size[0]}x{enhanced_image.size[1]} pixels")
        logger.info(f"File: {file_size:.1f} MB")
        logger.info(f"Source: Google Satellite")
        
        return filename
        
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None

def main_google_satellite_workflow():
    """Main workflow for getting Google Satellite imagery"""
    
    logger.info("Google Satellite Imagery Downloader")
    logger.info("=" * 50)
    logger.info("Using Google Satellite source only")
    
    # Get location
    location_data = get_user_location()
    
    if 'error' in location_data:
        logger.error(f"Location error: {location_data['error']}")
        
        # Fallback: ask for manual coordinates
        try:
            logger.info("Manual coordinate entry required:")
            lat = float(input("Enter latitude: "))
            lon = float(input("Enter longitude: "))
            location_data = {'latitude': lat, 'longitude': lon, 'city': 'Manual', 'region': 'Entry'}
        except ValueError:
            logger.error("Invalid coordinates provided.")
            return
    
    lat, lon = location_data['latitude'], location_data['longitude']
    logger.info(f"Location: {location_data.get('city', 'Unknown')}, {location_data.get('region', 'Unknown')}")
    logger.info(f"Coordinates: {lat:.6f}, {lon:.6f}")
    
    # Configuration for maximum quality
    zoom_level = 19  # Maximum zoom for Google
    tile_grid_size = 5  # 5x5 grid = 25 tiles
    
    logger.info(f"Configuration:")
    logger.info(f"Zoom level: {zoom_level} (maximum detail)")
    logger.info(f"Tile grid: {tile_grid_size}x{tile_grid_size} = {tile_grid_size**2} tiles")
    logger.info(f"Target resolution: ~{256 * tile_grid_size}x{256 * tile_grid_size} pixels")
    logger.info(f"Estimated time: 1-3 minutes")
    
    # Download Google Satellite imagery
    logger.info("Starting Google Satellite download...")
    start_time = time.time()
    
    image = get_google_satellite_imagery(lat, lon, zoom_level, tile_grid_size)
    
    download_time = time.time() - start_time
    logger.info(f"Download completed in {download_time:.1f} seconds")
    
    if not image:
        logger.error("No image was successfully downloaded.")
        logger.info("Try again later or check your internet connection.")
        return
    
    # Display results
    logger.info("SUCCESS! Downloaded Google Satellite image:")
    pixels = image.size[0] * image.size[1]
    logger.info(f"Google Satellite: {image.size[0]:,}x{image.size[1]:,} pixels ({pixels/1000000:.1f}M pixels)")
    
    # Save image
    logger.info("Saving enhanced image...")
    saved_file = save_satellite_image(image, lat, lon)
    
    # Display image
    logger.info("Displaying satellite image...")
    display_satellite_image(image, lat, lon)
    
    logger.info("COMPLETE!")
    if saved_file:
        logger.info(f"Google Satellite image saved as: {saved_file}")
    logger.info("High-resolution satellite imagery ready!")

if __name__ == "__main__":
    main_google_satellite_workflow()