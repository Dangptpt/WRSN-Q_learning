from sklearn.cluster import KMeans
import yaml
import numpy as np
import matplotlib.pyplot as plt

def kmeans_clustering(list_node, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(list_node)
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(list_node) 
    return centers

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

data = read_yaml_file(r"D:\Hust Study\Module\Module 3\Evolutionary Computation\WRSN\physical_env\network\network_scenarios\hanoi1000n200.yaml")
list_node = np.array(data['nodes'])

# Phân cụm bằng KMeans với đầu ra là 80 cụm
cluster_centers = kmeans_clustering(list_node, n_clusters=49)
cluster_centers = np.vstack((np.array([[500, 500]]), cluster_centers))
print(cluster_centers)