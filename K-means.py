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


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))#change to array
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(pts, ctd, dim):
    # Initialize to K random centers
    oldmu = pts[np.random.choice(int(pts.size/dim),ctd,replace=False)]#change to array
    mu = pts[np.random.choice(int(pts.size/dim),ctd,replace=False)]#change to array
    while not has_converged(mu, oldmu):
        print(mu)
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(pts, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


#plot if dimension = 2
def scatter_plot(pts, ctd):
    plt.scatter(pts[:, 0], pts[:, 1], color='blue')
    plt.scatter(ctd[:, 0], pts[:, 1], color='red')
    return plt.show()

#scatter_plot(generate_point_uniform(number_of_point,number_of_dimension))
points = generate_point_uniform(number_of_point,number_of_dimension)
A,B = find_centers(points,number_of_centroid,number_of_dimension)
#print(A,B)