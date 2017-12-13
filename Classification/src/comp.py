import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage : python3 comp.py <name1> <name2>")
        sys.exit()

    # Input
    data1 = sys.argv[1]
    data2 = sys.argv[2]

    d1 = np.loadtxt(open(data1, "rb"), delimiter=",", dtype=np.str)
    d2 = np.loadtxt(open(data2, "rb"), delimiter=",", dtype=np.str)

    cpt = 0
    for i in range(1, len(d1)):
        id = d1[i, 0]
        label = d1[i, 1]
        for j in range(1, len(d2)):
            if (d2[j, 0] == id):
                if (d2[j, 1] == label):
                    cpt += 1
                break;

    print("Ressemblance = ", cpt/len(d2))
