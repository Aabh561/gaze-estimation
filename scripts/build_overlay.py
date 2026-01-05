import cv2
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
OVERLAYS = os.path.join(ROOT, 'overlays')
OUT_PATH = os.path.join(ROOT, 'overlay.jpg')

paths = [
    os.path.join(OVERLAYS, 'WIN_20251203_01_12_35_Pro_overlay.jpg'),
    os.path.join(OVERLAYS, 'WIN_20251204_22_08_34_Pro_overlay.jpg'),
    os.path.join(OVERLAYS, 'WIN_20251204_22_08_35_Pro_overlay.jpg'),
    os.path.join(OVERLAYS, 'WIN_20251204_22_08_42_Pro_overlay.jpg'),
]

imgs = []
for p in paths:
    img = cv2.imread(p)
    if img is None:
        raise SystemExit(f'Failed to read {p}. Ensure the per-image overlays exist.')
    imgs.append(img)

# Determine tile size (smallest among inputs)
hmin = min(i.shape[0] for i in imgs)
wmin = min(i.shape[1] for i in imgs)

# Normalize tile size
tiles = [cv2.resize(i, (wmin, hmin)) for i in imgs]

# Add simple filename labels bar at top
for tile, p in zip(tiles, paths):
    name = os.path.basename(p).replace('_overlay.jpg','')
    cv2.rectangle(tile, (0,0), (tile.shape[1], 36), (0,0,0), -1)
    cv2.putText(tile, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# Assemble 2x2 grid
row1 = cv2.hconcat(tiles[0:2])
row2 = cv2.hconcat(tiles[2:4])
grid = cv2.vconcat([row1, row2])

# Optional border
border = 8
grid = cv2.copyMakeBorder(grid, border, border, border, border, cv2.BORDER_CONSTANT, value=(0,0,0))

cv2.imwrite(OUT_PATH, grid)
print('Wrote', OUT_PATH, 'size', grid.shape)