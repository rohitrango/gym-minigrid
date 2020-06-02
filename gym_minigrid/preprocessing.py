from gym_minigrid import minigrid
from skimage import measure
import numpy as np
from matplotlib import pyplot as plt

def preprocess_connected_components(nmap, index_mapping):
    '''
    nmap: numpy map
    index_mapping: mapping from int to key
    '''
    conn = nmap * 0 + 1  # Check for connectivity map (make all walls 0)
    wallidx = None
    print(index_mapping)
    for k, v in index_mapping.items():
        print(v)
        if v == 'wall':
            wallidx = k
            wallmap = 1 - (nmap == k).astype(int)
            conn = conn * wallmap
        elif v == 'unseen':
            wallmap = 1 - (nmap == k).astype(int)
            conn = conn * wallmap

    # get labels
    labels = measure.label(conn, background=0, connectivity=1)
    #plt.figure()
    #plt.imshow(labels.T)
    #plt.show()

    M = np.max(labels) + 1
    arr = []
    for i in range(1, M):
        count = len(np.where(labels == i)[0])
        arr.append(count)

    # Get argmax and replace all of them with wall
    argmax = np.argmax(arr) + 1
    for i in range(1, M):
        if i == argmax:
            continue
        # Not argmax, replace the indices
        y, x = np.where(labels == i)
        nmap[y, x] = wallidx

    return nmap
