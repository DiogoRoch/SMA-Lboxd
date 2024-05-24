# SMA-Lboxd
## Repository with all code for the processing, analysis, exploration, community detection and evaluation of our Lboxd network and community detection algorithm

### Disclaimer
- **Python version used was mainly 3.11.8** and notebooks were mainly run with Visual Studio Code's jupyter extension.
- Our project was done mostly through jupyter notebooks, all of which are provided through this github repository, you can also find most datasets used in the Data folder.
- The database we used to augment our network could not be compressed at a size low enough to add it to this github.
    - Download movies.csv at https://www.kaggle.com/datasets/gsimonx37/letterboxd
    - genres.csv and countries.csv however are inside the Data folder.
### **Data folder**
- Our augmented dataset is stored as a zip file inside the Data folder "augm_lboxd.zip":
- **augm_lboxd.csv** -> contains our augmented dataset (users and movies + date, duration, genre, country, average rating)
    - This dataset should also be produced when running notebook 4.
- **lbxd_genres.csv** -> contains the genres corresponding to movies.csv.
- **lbxd_countries.csv** -> contains the countries corresponding to movies.csv.
- **augmented_movies.csv** -> contains augmentations for all movies in our dataset (just the movies, no users for compression purposes)
- **lboxd_IDcoded.csv** -> our clean dataset without augmentations and with an addition ID column with a unique ID for each unique movie title.
- **lboxd_no_single_IDcoded.csv** -> version of lboxd_IDcoded.csv where we removed all movies with only 1 rating as a sampling method.
- **letterbox_anonym** -> this is the original dataset downloaded from kaggle at https://www.kaggle.com/datasets/rbertvmosi/letterboxd-ratings-14m
- **m100_u200.csv** -> this is another type of sample where we only kept the top 100 movies and top 200 users.
- **moviesR_usersA.csv** -> a sample where we took a random set of 500 movies and all users associated to them.
- **augm_movies400.csv** -> a sample containing only movies with more than 400 ratings and all users that have rated them.

## **Structure of the notebooks**
- In order to run our code used for our project, all notebooks we used are proposed in a numbered form which represents the order in which they are to be run.

### **Data Processing [Diogo Rocha]**
- **Notebook 1**: Data_Processing
    - First exploration of the dataset.
    - This notebook cleans the data of missing values, casts the columns to their correct data types, codes each unique movie to a specific ID and adds those IDs as an additional column.
    - This notebook also makes a first sample of the dataset without movies that were rated only once.
- **Notebook 4**: Data_Processing2_And_Plotting
    - This notebook analyzes the missing values a little more.
    - It also produces different types of sampling:
        - Sampling by iteratively removing movies having less than a certain amount of ratings.
        - Sampling by randomly selecting a certain amount of movies.
        - Sampling by taking the top 200 users and top 100 movies (can also be expanded to take variable amounts of top users/movies)

### **Data Augmentation [Diogo Rocha]**
- **Notebook 2**: Data_Augmentation
    - Exploration of multiple databases (ended up only using movies.csv because the other one did not have many relevant movies while being extremely heavy)
    - Creation of the augmented_movies.csv that contains the movies from our original dataset with their respective augmentations (date, minute "duration", mean rating, genre, country)
- **Notebook 3**: Data_Augmentation2
    - This notebook's role is to use the augmented_movies.csv and our lboxd_IDcoded.csv to merge them and produce an augmented dataset (that is stored in a zip file or can be obtained by running this notebook)

### **Network Exploration [Diogo Rocha]**
- **Notebook 5**: Network_Exploration
    - This notebook computes the degree centrality and betweenness centrality of our dataset.
    - It also observes the top 10 users' and movies' depdending on DC and BC, and the average DC and BC.
    - The genre and country distribution is studied for these top movies and the plots are used in the report.
    - RUNTIME: degree centrality runs very fast.
    - RUNTIME: betweenness centrality took around 20 minutes to run on my macbook air M2 8gb RAM, 256gb SSD

### **Community Detection Algorithm [Hannah Portmann]**
- **biloubvain_module.py**:
    - This is our implementation of the Bilouvain algorithm.
- **Notebook 6**: Testing_And_Visualization
    - In the first part of this notebook, there is code which shows how the Graph has to be prepared to be given to our algorithm and how our algorithm can be run on specific data.
    - To run the algorithm with a different dataset, these steps should always be followed to ensure correct functioning of the algorithm.

### **Network Visualization [Hannah Portmann]**
- **Notebook 6**: Testing_And_Visualization
    - In the second part of this notebook, the Graph is visualize, once with the nodes colored as their node types and once as their communities.
    - This was done with to different data sample sets:
        - RUNTIME for 'moviesR_usersA.csv': quite fast, a couple of minutes
        - RUNTIME for 'augm_movies400.csv': longer, around one hour

### **Evaluation [Zeynep Sema Aydin]**
- **Notebook 7.1**
    - This notebook consist of the main algorithm implementation. Data is randomly sampled, the sample with the optimal runtime - community quality is chosen, then the modularity value for that sample is calculated using sknetwork, with constructing the biadjacency matrix and obtaining the community labels for nodes. Then the results for chosen dataset is compared with results from other algorithms. It takes approx. 15 mins. to run this notebook if scikit-network is already installed.
- **Notebook 7.2**
    - This notebook is for getting the results for the whole dataset from comparison algorithms. It takes about a minute- a minute and a half to run if scikit-network is already installed. 


## Teacher Recommendations
### If you want to run everything
- Ensure that you download the letterboxd database "movies.csv" from the kaggle link provided on the top of the readme file.
- Follow the order of the notebooks, multiple analysis and our logic is inside.
### Focus on the main algorithm and evaluation
- Ensure that you unzip our augmented dataset.
- If you want to look at our network exploration, then run notebook 5.
- If you want to look at our implementation of a bilouvain community detection algorithm, then run the bilouvain notebook. The bilouvain implementation is stored inside bilouvain_module.py and used by the notebook.
- If you want to look at the evaluation then run the evaluation notebooks.

