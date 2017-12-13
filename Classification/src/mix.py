import sys
import numpy as np
import random

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage : python3 mix.py <File1> <File2> <File3> <File4> <File5> <out>")
        sys.exit()



    d1 = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", dtype=np.str)
    d2 = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=np.str)
    d3 = np.loadtxt(open(sys.argv[3], "rb"), delimiter=",", dtype=np.str)
    d4 = np.loadtxt(open(sys.argv[4], "rb"), delimiter=",", dtype=np.str)
    d5 = np.loadtxt(open(sys.argv[5], "rb"), delimiter=",", dtype=np.str)

    cpt = 0
    rr = []
    for i in range(1, len(d1)):
        r = [0, 0, 0, 0, 0, 0, 0, 0]
        r[int(d1[i, 1])-1] += 1
        r[int(d2[i, 1])-1] += 1
        r[int(d3[i, 1])-1] += 1
        r[int(d4[i, 1])-1] += 1
        r[int(d5[i, 1])-1] += 1
        rr.append(r)
        print(r)

    res = []
    res.append(['track_id', 'genre_id'])

    for i in range(len(d1)-1):
            id = d1[i+1, 0]
            label = 0;
            max = 0;
            ll = []
            good = False
            for j in range(8):
                if rr[i][j]!= 0:
                    ll.append(j+1)

                if rr[i][j] >= 4:
                    max = rr[i][j]
                    label = j+1
                    good = True
                    break


            if good == False:
                nombreDeBase = random.randint(0, len(ll) -1)
                label = ll[nombreDeBase]

            res.append([id, label])

    np.savetxt(sys.argv[6], res, delimiter=",", fmt="%s")
