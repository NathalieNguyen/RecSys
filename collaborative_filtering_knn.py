from lib.preprocessing import data_for_training as data
from lib.evaluation import recommendations as recs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    np.random.seed(0)

    stratify_df = data.filter_ratings_above_count_threshold(frac=0.1)
    train_set, test_set = train_test_split(stratify_df, stratify=stratify_df['User-ID'])

    IxU_train = train_set.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
    IxU_train_sparse = csr_matrix(train_set.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0))
    IxU_test_sparse = csr_matrix(test_set.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0))

    item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    item_knn.fit(IxU_train_sparse)

    query_index = np.random.choice(IxU_train.shape[0])
    distances, indices = item_knn.kneighbors(IxU_train.iloc[query_index, :].values.reshape(1, -1), n_neighbors=10)

    print(distances)
    print(indices)

    isbn_book_dict = recs.map_isbn_to_names()

    print('Recommendations for {0}, ISBN: {1}:\n'.format(isbn_book_dict[IxU_train.index[query_index]], IxU_train.index[query_index]))

    for i in range(0, len(distances.flatten())):
        print('{0}: {1}, ISBN: {2} with distance of {3}:'.format(i, isbn_book_dict[IxU_train.index[indices.flatten()[i]]],
                                                                 IxU_train.index[indices.flatten()[i]], distances.flatten()[i]))
