import os
os.environ["PROJ_LIB"] = "C:\\Users\\97252\\Miniconda3\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
bm = Basemap()
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import itertools
import requests
import json
# call the OSMR API
def dist(lon_lat1, lon_lat2):
    r = requests.get(f"http://router.project-osrm.org/route/v1/car/{lon_lat1[1]},{lon_lat1[0]};{lon_lat2[1]},{lon_lat2[0]}?overview=false""")
    # then you load the response using the json libray
    # by default you get only one alternative so you access 0-th element of the `routes`
    routes = json.loads(r.content)
    route_1 = routes.get("routes")[0].get("legs")[0].get("distance")
    return route_1

# then cast your geographic coordinate pair to the projected system
def calculateDistInKLonLat(first, sec):
    # approximate radius of earth in km
    R = 6373.0
    dlon = radians(first[0]) - radians(sec[0])
    dlat = radians(first[1]) - radians(sec[1])
    a = sin(dlat / 2) ** 2 + cos(first[1]) * cos(sec[1]) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
def main():
    mu = np.array([32.085113513216214, 34.80001533158019,9,17]) # lon, lat
    cov = np.array([[0.2, 0,0,0], [0, 0.03,0,0],[0,0,3,0], [0,0,0,3]])
    X = np.random.multivariate_normal(mu, cov, 500)
    for tup in X:
        if not bm.is_land(tup[1], tup[0]):
            tup[2] = 0
    ind = np.where(X.T[2] != 0)
    X = X[ind]
    n = len(ind[0])
    sample_ids = np.arange(0, n)
    all_distances = np.zeros((n,n))
    for i in range(n):
        all_distances[i,i] = 0
    j = 0
    print("total: "+str(n))
    for comb in itertools.combinations(sample_ids, 2):
        j +=1
        print(j)
        #dist = calculateDistInKLonLat(X[comb[0], :2], X[comb[1], :2])
        d = dist(X[comb[0], :2], X[comb[1], :2])
        all_distances[comb[0], comb[1]] = d/1000
        all_distances[comb[1], comb[0]] = d/1000
        if d/1000 > 200:
            print(X[comb[0], :2])
            print(X[comb[1], :2])
            print(d/1000)
    print(all_distances)
    np.savetxt("all_distances_csv.csv", all_distances, delimiter=",")
    np.save("all_distances_npy.npy", all_distances)
    np.save("X.npy", X)

main()
def analyze():
    work_lon_lat = np.array([32.085113513216214, 34.80001533158019])
    distances = np.load("all_distances_npy.npy")
    X = np.load("X.npy")
    distances_from_work = list()
    for i in range(X.shape[0]):
        distances_from_work.append(dist(X[i, :2], work_lon_lat))
    # triu_ind = np.triu_indices(distances.shape[0],1)
    # upper_triangle = distances[triu_ind]
    sorted_by_arrival_time = X[np.argsort(X.T[2])]
    filled_cars = list()
    index = 0
    everyone = set(np.arange(0, X.shape[0]))
    #while index < X.shape[0]:
    while len(everyone) > 0:
        if sorted_by_arrival_time[index] not in everyone:
            index += 1
            continue
        filled_cars.append(list())
        filled_cars[X.shape[0]-1].append(sorted_by_arrival_time[index])
        everyone.remove(sorted_by_arrival_time[index])
        passengers = 0
        driver_index = index
        driver = sorted_by_arrival_time[index]
        while passengers < 2:
            index += 1
            if sorted_by_arrival_time[index] not in everyone:
                index += 1
                continue
            if (distances[sorted_by_arrival_time[driver]][sorted_by_arrival_time[index]]/60 + distances_from_work[sorted_by_arrival_time[index]]/60)-distances_from_work[driver]<=0.34:
                        filled_cars[X.shape[0] - 1].append(sorted_by_arrival_time[index])
                        everyone.remove(sorted_by_arrival_time[index])
                        passengers += 1
        index = driver_index + 1













