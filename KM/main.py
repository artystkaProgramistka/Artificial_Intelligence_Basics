# This is a program for clustering using K-Means algorithm

from random import sample
import matplotlib.pyplot as plt


def read_file(path):
    train = open(path)
    lines = []
    for line in train:
        lines.append(list(map(float, line.split(','))))
    return lines


def compute_nearest_centriod(centroids, observation):
    pairs = []  # distance - centroid_id
    for centroid_id, centroid in enumerate(centroids):
        distance = 0
        for i in range(len(observation)):
            distance += (observation[i] - centroid[i])**2
        pairs.append((distance, centroid_id))
    return min(pairs)[1]

def generate_clusters(centroids, data):
    centroid_observations = dict()
    for centroid_id in range(len(centroids)):
        centroid_observations[centroid_id] = []
    for observation in data:
        centroid_observations[compute_nearest_centriod(centroids, observation)].append(observation)
    return centroid_observations


def center_centriods(centroids, centroid_observations):
    dimensions = len(centroids[0])
    new_centroids = []
    for centroid_id, centroid in enumerate(centroids):
        # how to calculate the center between the points?
        new_centroid=[]
        for dimension in range(dimensions):
            sum = 0
            for observation in centroid_observations[centroid_id]:
                sum += observation[dimension]
            average = sum / dimensions
            new_centroid.append(average)
        new_centroids.append(new_centroid)
    if centroids == new_centroids:
        return True, new_centroids
    else:
        return False, new_centroids


def KM(file, k, max_iters=1000):
    data = read_file(file)
    centroids = init_centroids(data, k)
    assert centroids
    centroid_observations = generate_clusters(centroids, data)
    return centroids, centroid_observations
    for _ in range(max_iters):
        pair = center_centriods(centroids, centroid_observations)
        if pair[0]: break
        centroids = pair[1]
        assert centroids
        centroid_observations = generate_clusters(centroids, data)
    return centroids, centroid_observations


def init_centroids(data, k):
    return sample(data, k)


def optimize(file, min_k, max_k, k_cost, accuracy_gain):
    max = float('inf')
    best_k = -1
    results = []
    for k in range(min_k, max_k + 1):
        centroids, centroids_observations = KM(file, k)
        avs_of_distances = 0
        for centroid_id, centroid in enumerate(centroids):
            sum_of_distances = 0
            for observation in centroids_observations[centroid_id]:
                distance = 0
                for dimension in range(len(centroids[0])):
                    distance += (observation[dimension] - centroid[dimension])**2
                sum_of_distances += distance
            avs_of_distances += sum_of_distances / len(centroids_observations[centroid_id])
        results.append(avs_of_distances)
        result = avs_of_distances / accuracy_gain + k * k_cost
        if result < max:
            max = result
            best_k = k
    # draw a plot
    plt.figure(figsize=(20, 10))
    plt.title('Average distances of points from their centroids for various k')
    plt.plot(list(range(min_k, max_k+1)), results)
    plt.xlabel('k')
    plt.ylabel('Average distance')
    plt.grid()
    plt.show()
    return best_k


def main():
    k = optimize('test.txt', 1, 30, 1, 1)
    centroids, centroids_observations = KM('test.txt', k)
    for centroid_id, centroid in enumerate(centroids_observations.keys()):
        print('centroid ', centroid_id, ':')
        for observation in centroids_observations[centroid]:
            print('\t', observation)

if __name__ == '__main__':
    main()