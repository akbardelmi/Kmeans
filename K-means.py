#import package

import numpy as np
import matplotlib.pyplot as plt

#input total centroid

#number_of_point = 1000
number_of_point = int(input("press input total number of points : "))
number_of_centroid = int(input("press input total number of centroid : "))
number_of_dimension = 2

#generating random points
def generate_point_uniform(nb_point,dim):
    pts = np.zeros((nb_point,dim))
    for i in range(nb_point):
        pts[i] = np.random.rand(1, dim)
    return pts


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]#?/?????
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(centroid, clusters):
    new_centroid = np.zeros(centroid.shape)
    keys = sorted(clusters.keys())
    for k in keys:
        new_centroid[k, :] = np.mean(clusters[k], axis=0)
    return new_centroid

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(pts, ctd, dim):
    # Initialize to K random centers
    centroid = np.zeros((ctd, dim))
    old_centroid = np.zeros((ctd, dim))
    shuffle = np.random.choice(int(pts.size/dim), ctd, replace=False)
    for i in range(ctd):
        centroid[i, :] = pts[shuffle[i], :]
    while not has_converged(centroid, old_centroid):
        print(centroid)
        scatter_plot(pts, centroid)
        old_centroid = centroid
        # Assign all points in X to clusters
        clusters = cluster_points(pts, centroid)
        # Reevaluate centers
        centroid = reevaluate_centers(old_centroid, clusters)
    return(centroid, clusters)


#plot if dimension = 2
def scatter_plot(pts, centroid):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(pts[:, 0], pts[:, 1], c='b')
    ax1.scatter(centroid[:, 0], centroid[:, 1], c='r', marker='*')
    return plt.show()

#scatter_plot(generate_point_uniform(number_of_point, number_of_dimension))
points = generate_point_uniform(number_of_point,number_of_dimension)
A,B = find_centers(points,number_of_centroid,number_of_dimension)
#print(A,B)