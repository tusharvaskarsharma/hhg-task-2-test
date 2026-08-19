class Index:
    def __init__(self, space, dim):
        pass
    def load_index(self, path):
        pass
    def set_ef(self, ef):
        pass
    def get_current_count(self):
        return 100000
    def knn_query(self, query_vector, k=10):
        import numpy as np
        # Return dummy labels and distances
        labels = np.arange(k).reshape(1, -1)
        distances = np.zeros((1, k))
        return labels, distances
