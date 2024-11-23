import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

csv_path = "../datasets/Country-data.csv"
csv_df = pd.read_csv(csv_path)
array = csv_df.values
array_trimmed = array[:, 1:]
x_fitted = array_trimmed
kmeans_kwargs = {
    "init": "random",
    "n_init": "auto",
    "max_iter": 500,
    "random_state": 10,
}

def get_kmeans_cluster_num():
    sse = []
    testing_range = range(1, 17)
    for k in testing_range:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x_fitted)
        print(kmeans.inertia_)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(
        testing_range, sse, curve="convex", direction="decreasing"
    )

    print(f"Elbow Point: {kl.elbow}")
    kl.plot_knee()
    plt.show()
    return kl.elbow

def analyse_data():
    # get ideal cluster num
    k_num = get_kmeans_cluster_num()

    # fit data
    kmeans = KMeans(n_clusters=k_num, **kmeans_kwargs)
    kmeans.fit(x_fitted)

    # merge cluster labels to data frame
    csv_df["class"] = pd.Series(kmeans.labels_).values

    # print centroids for comparison
    column_names = csv_df.columns.tolist()[1:-1]
    centroids = np.unstack(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=column_names)
    print(centroids_df)

analyse_data()