import numpy as np
import pystac_client
import planetary_computer
import rasterio

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

catalog = pystac_client.Client.open(CATALOG_URL)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-125.0, 47.0, -121.0, 50.0],
    datetime="2024-06-01/2024-06-30",
    query={"eo:cloud_cover": {"lt": 20}},
    limit=1,
)

item = next(search.items())
print("Scene ID:", item.id)

band_keys = ["B02", "B03", "B04", "B08"]
signed_assets = {
    key: planetary_computer.sign(item.assets[key].href)
    for key in band_keys
}

arrays = {}

for key, url in signed_assets.items():
    with rasterio.open(url) as src:
        arr = src.read(1).astype(np.float32)
        arrays[key] = arr
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

b2 = arrays["B02"]
b3 = arrays["B03"]
b4 = arrays["B04"]
b8 = arrays["B08"]

eps = 1e-6

chlorophyll = b3 / (b2 + eps)
turbidity = b4 / (b3 + eps)
ndwi = (b3 - b8) / (b3 + b8 + eps)

water_mask = ndwi > 0

# no real temperature yet
temperature_proxy = np.zeros_like(chlorophyll, dtype=np.float32)

risk = (
    0.5 * chlorophyll +
    0.3 * turbidity +
    0.2 * temperature_proxy
)

risk = np.where(water_mask, risk, np.nan)

risk_min = np.nanmin(risk)
risk_max = np.nanmax(risk)
risk_norm = (risk - risk_min) / (risk_max - risk_min + eps)

print("Risk min:", float(np.nanmin(risk_norm)))
print("Risk max:", float(np.nanmax(risk_norm)))
print("Water pixels:", int(np.sum(water_mask)))