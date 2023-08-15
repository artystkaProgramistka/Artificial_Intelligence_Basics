# Title: K-Means Clustering with Optimization and Visualization

# Description:
# This Python program implements a simple, yet effective, K-means clustering algorithm with optimization, visualization of the clusters, and a confusion matrix. It takes a CSV file containing multi-dimensional data points with an optional label, applies the K-means algorithm to cluster the data, and visualizes the optimal number of clusters (k) using the elbow method. The algorithm converges to a solution when the centroids no longer move or after a specified maximum number of iterations. The program is built using object-oriented programming principles and can be easily extended or modified.

# Features:

# 1. Read data from a CSV file containing features and optional labels.
# 2. Flexible data input: handles multi-dimensional data with any number of features.
# 3. Implementation of the K-means algorithm with customizable maximum iterations.
# 4. Optimal k value search using the elbow method with customizable parameters.
# 5. Visualization of the average distances of points from their centroids for various k values.
# 6. Calculation of cluster clearness based on the labels (if provided).
# 7. Object-oriented design for easy modification and extension.
# 8. 2D visualization of the clustered data points using PCA.
# 9. Generation of a confusion matrix to compare the true labels with the predicted clusters.

# Usage:

# 1. Instantiate the KMeans class with the desired input file, number of clusters (k), and maximum iterations (optional).
# 2. Call the `fit()` method on the KMeans instance to perform clustering.
# 3. (Optional) Use the `optimize()` method to find the optimal k value.
# 4. Review the output, including cluster centroids, observations, and clearness.
# 5. Call the `visualize_clusters()` method to display a 2D scatter plot of the clustered data points.
# 6. Call the `confusion_matrix()` method to display a heatmap of the confusion matrix comparing the true labels with the predicted clusters.


from random import sample
from sklearn.decomposition import PCA
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Observation:
    def __init__(self, features, label=None):
        self.features = features
        self.label = label

class KMeans:
    def __init__(self, file, k, max_iters=1_000):
        self.file = file
        self.k = k
        self.observations = self.read_file()
        self.centroids = self.init_centroids()
        self.centroid_observations = self.generate_clusters()
        self.max_iters = max_iters

    def read_file(self):
        data = open(self.file)
        observations = []
        for line in data:
            if not line.strip(): continue
            *features, label = line.split(',')
            features = list(map(float, features))
            observations.append(Observation(features, label))
        return observations

    def init_centroids(self):
        return [obs.features for obs in sample(self.observations, self.k)]


    def compute_nearest_centroid(self, observation):
        pairs = []  # distance - centroid_id
        for centroid_id, centroid in enumerate(self.centroids):
            distance = 0
            for i in range(len(centroid)):
                distance += (observation.features[i] - centroid[i]) ** 2
            pairs.append((distance, centroid_id))
        return min(pairs)[1]

    def generate_clusters(self):
        centroid_observations = dict()
        for centroid_id in range(len(self.centroids)):
            centroid_observations[centroid_id] = []
        for observation in self.observations:
            centroid_observations[self.compute_nearest_centroid(observation)].append(observation)
        return centroid_observations

    def center_centroids(self):
        dimensions = len(self.centroids[0])
        new_centroids = []

        for centroid_id, centroid in enumerate(self.centroids):
            new_centroid = []
            num_observations = len(self.centroid_observations[centroid_id])

            if num_observations > 0:
                for dimension in range(dimensions):
                    total = 0
                    for observation in self.centroid_observations[centroid_id]:
                        total += observation.features[dimension]
                    average = total / num_observations
                    new_centroid.append(average)
            else:
                new_centroid = centroid  # Keep the current centroid if there are no observations

            new_centroids.append(new_centroid)

        if self.centroids == new_centroids:
            return True, new_centroids
        else:
            return False, new_centroids

    def print_iteration_metrics(self):
        total_sum_of_distances = 0
        print("Clearness of each cluster:")

        for centroid_id, centroid in enumerate(self.centroids):
            sum_of_distances = 0
            label_count = {}

            for observation in self.centroid_observations[centroid_id]:
                distance = 0
                for dimension in range(len(centroid)):
                    distance += (observation.features[dimension] - centroid[dimension]) ** 2
                sum_of_distances += distance

                # Count the labels in the current cluster
                if observation.label in label_count:
                    label_count[observation.label] += 1
                else:
                    label_count[observation.label] = 1

            total_sum_of_distances += sum_of_distances
            clearness = 0
            if len(self.centroid_observations[centroid_id]) > 0:
                most_common_label_count = max(label_count.values())
                clearness = (most_common_label_count / len(self.centroid_observations[centroid_id])) * 100

            print(f"Cluster {centroid_id}: {clearness:.2f}%")

            print(f"Sum of distances between the observations and the centroids: {total_sum_of_distances}")

    def fit(self):
        self.observations = self.read_file()
        self.centroids = self.init_centroids()
        assert self.centroids
        self.centroid_observations = self.generate_clusters()
        for _ in range(self.max_iters):
            is_converged, new_centroids = self.center_centroids()
            if is_converged:
                break
            self.centroids = new_centroids
            assert self.centroids
            self.centroid_observations = self.generate_clusters()
            # if self.print_metrics:
            self.print_iteration_metrics()
        return self.centroids, self.centroid_observations

    def optimize(self):
        max_value = float('inf')
        best_k = -1
        results = []

        for k in range(self.min_k, self.max_k + 1):
            kmeans_instance = KMeans(file=self.file, k=k)
            centroids, centroid_observations = kmeans_instance.fit()
            avs_of_distances = 0

            for centroid_id, centroid in enumerate(centroids):
                sum_of_distances = 0
                num_observations = len(centroid_observations[centroid_id])

                if num_observations > 0:
                    for observation in centroid_observations[centroid_id]:
                        distance = 0
                        for dimension in range(len(centroid)):
                            distance += (observation.features[dimension] - centroid[dimension]) ** 2
                        sum_of_distances += distance
                    avs_of_distances += sum_of_distances / num_observations

            results.append(avs_of_distances)
            result = avs_of_distances / self.accuracy_gain + k * self.k_cost

            if result < max_value:
                max_value = result
                best_k = k

        # Plot the results
        plt.figure(figsize=(20, 10))
        plt.title('Average distances of points from their centroids for various k')
        plt.plot(list(range(self.min_k, self.max_k + 1)), results)
        plt.xlabel('k')
        plt.ylabel('Average distance')
        plt.grid()
        plt.show()

        return best_k

    def visualize_clusters(self):
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform([obs.features for obs in self.observations])

        plt.figure(figsize=(10, 7))
        plt.title("K-Means Clustering (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        # Plot observations
        colors = plt.cm.rainbow(np.linspace(0, 1, self.k))
        for centroid_id, observations in self.centroid_observations.items():
            x = [reduced_features[self.observations.index(obs)][0] for obs in observations]
            y = [reduced_features[self.observations.index(obs)][1] for obs in observations]
            plt.scatter(x, y, c=[colors[centroid_id]], label=f"Cluster {centroid_id}")

        # Plot centroids
        reduced_centroids = pca.transform(self.centroids)
        centroids_x = [centroid[0] for centroid in reduced_centroids]
        centroids_y = [centroid[1] for centroid in reduced_centroids]
        plt.scatter(centroids_x, centroids_y, c="black", marker="x", s=100, label="Centroids")

        plt.legend()
        plt.grid()
        plt.show()

    def confusion_matrix(self):
        labels = sorted(list(set(obs.label for obs in self.observations)))
        matrix = defaultdict(lambda: defaultdict(int))

        for centroid_id, observations in self.centroid_observations.items():
            for observation in observations:
                matrix[observation.label][centroid_id] += 1

        print("Confusion Matrix:")
        print("\t", end="")
        for centroid_id in range(self.k):
            print(f"Cluster {centroid_id}", end="\t")
        print()

        for label in labels:
            print(label, end="\t")
            for centroid_id in range(self.k):
                print(matrix[label][centroid_id], end="\t")
            print()

        return matrix

    def confusion_matrix(self):
        labels = sorted(list(set(obs.label for obs in self.observations)))
        matrix = defaultdict(lambda: defaultdict(int))

        for centroid_id, observations in self.centroid_observations.items():
            for observation in observations:
                matrix[observation.label][centroid_id] += 1

        # Convert the defaultdict to a 2D numpy array
        matrix_array = np.zeros((len(labels), self.k))
        for i, label in enumerate(labels):
            for j, centroid_id in enumerate(range(self.k)):
                matrix_array[i, j] = matrix[label][centroid_id]

        # Cast the elements of matrix_array to integers
        matrix_array = np.vectorize(int)(matrix_array)

        # Plot the confusion matrix using seaborn heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix_array, annot=True, cmap="YlGnBu", fmt="d", xticklabels=[f"Cluster {i}" for i in range(self.k)], yticklabels=labels)
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        return matrix

def main():
    k = 2
    kmeans = KMeans(file='data.txt', k=k)
    centroids, centroid_observations = kmeans.fit()
    for centroid_id, centroid in enumerate(centroid_observations.keys()):
        print('centroid ', centroid_id, ':')
        for observation in centroid_observations[centroid]:
            print(f'\t{observation.features} {observation.label}')
    kmeans.visualize_clusters()
    kmeans.confusion_matrix()

if __name__ == '__main__':
    main()

