import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-125.0, 47.0, -121.0, 50.0],   # Pacific Northwest example
    datetime="2024-06-01/2024-06-30",
    query={"eo:cloud_cover": {"lt": 20}},
    limit=3,
)

items = list(search.items())
print(f"Found {len(items)} items")

if not items:
    raise SystemExit("No scenes found")

item = items[0]
print("Scene ID:", item.id)

signed = planetary_computer.sign(item.assets["B04"].href)
print("Signed B04 URL:", signed)