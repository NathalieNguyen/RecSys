# A Recommender System for Book Crossing

The goal of this project is to build a recommender system for book crossing by predicting the user’s preferences. The recommender system explores the relationships between users and items with the result of creating a top-N recommendation list for a specific user. 


## Dataset

In the following the main data sources for this project are listed. The book genres used for content-based models are not available in the original dataset and were crawled from the corresponding book crossing website.

[Book Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)<br>
[Book Crossing Website](http://www.bookcrossing.com)


## Requirements & Installation

The required modules can be installed via the following commands:

```pip install Cython --install-option="--no-cython-compile"```<br>
```pip install -r requirements.txt```

For installing PyTorch follow the instructions under the link below.<br>
[PyTorch](https://pytorch.org/get-started/locally/)


## Directory Structure

```
├── README.md
├── RecSys_Book_Crossing.ipynb              # Notebook presentation of the whole project
├── collaborative_filtering_knn.py
├── content_based.py
├── lib
│   ├── categories_scraper                  # Scraper for book categories
│   ├── data
│   │   ├── BX-CSV-Dump                     # Book crossing datasets
│   │   │   ├── BX-Book-Ratings.csv
│   │   │   ├── BX-Books.csv
│   │   │   └── BX-Users.csv
│   │   ├── ISBN.csv                        # ISBN used for scraper
│   │   ├── books_categories.json           # Scraped book categories
│   │   └── images
│   ├── evaluation
│   │   └── recommendations.py              # Dict with ISBN and book title
│   ├── features
│   │   └── categories.py                   # Merging book metadata with categories
│   ├── models
│   │   └── popularity_based.py
│   └── preprocessing                       # Loading and cleaning data
│       ├── books.py
│       ├── data_for_training.py            # Data preparation for training
│       ├── ratings.py
│       └── users.py
├── matrix_factorization_nn.py
├── matrix_factorization_svd.py
├── neural_network.py
├── random_predictions.py
└── requirements.txt
```
