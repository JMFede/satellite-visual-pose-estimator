import numpy as np

def createHeatmap(landmark, vp, heatmap_width, heatmap_height, sigma=1):
    heatmap = np.zeros((heatmap_height + 3, heatmap_width + 3))
    x, y = landmark
    if vp:
        for i in range(y - 3*sigma, y + 3*sigma):
            for j in range(x - 3*sigma, x + 3*sigma):
                heatmap[i, j] += np.exp(-((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2))
                
    heatmap = heatmap[1:-2, 1:-2]
        
    return heatmap

def coord2Heatmap(landmarks, visibility, heatmap_width=512, heatmap_height=512, sigma=1):
    heatmaps = []

    for landmark, vp in zip(landmarks, visibility):
        heatmaps.append(createHeatmap(landmark, vp, heatmap_width, heatmap_height, sigma))

    heatmaps = np.array(heatmaps).squeeze()

    return heatmaps