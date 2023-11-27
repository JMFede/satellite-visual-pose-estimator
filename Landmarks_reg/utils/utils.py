import numpy as np

def get_landmarks(landmarks_images, threshold1=-0.5, threshold2=-0.6):
    landmarks2D = []
    for img in landmarks_images:
        Lnd_found = False
        lndx = -1
        lndy = -1
        mask1 = img < threshold2
        img[mask1] = -1
        
        if np.max(img) > threshold1:
            Lnd_found = True
        elif np.max(img) > threshold2 and np.var(img) > 2e-6:
            Lnd_found = True
            
        if Lnd_found:
            lndy, lndx = np.where(img == np.max(img))
            lndy, lndx = np.round(np.mean(lndy)+1), np.round(np.mean(lndx)+1)
            if landmarks2D is not None or len(landmarks2D) > 0:
                for data in landmarks2D:
                    if data[2] == 1 and abs(lndx - data[0]) <= 3 and abs(lndy - data[1]) <= 3:
                        if np.max(img) > data[3] and np.var(img) > data[4]:
                            data[0] = -1
                            data[1] = -1
                            data[2] = 0
                            new_landmark = [lndx, lndy, 1, np.max(img), np.var(img)]
                        else:
                            new_landmark = [-1, -1, 0, np.max(img), np.var(img)]

            new_landmark = [lndx, lndy, 1, np.max(img), np.var(img)]
        else:
            new_landmark = [-1, -1, 0, np.max(img), np.var(img)]
        
        landmarks2D.append(new_landmark)

    return np.array(landmarks2D)