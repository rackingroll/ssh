SSH: (Sketch, Shingle, & Hash) for Indexing Massive-Scale Time Series
============

SSH is a time series indexing scheme. It allows you to approximately indexing the time series data. For the details SSH scheme, please refer the paper bellow:

Chen Luo and Anshumali Shrivastava "SSH (Sketch, Shingle, & Hash) for Indexing Massive-Scale Time Series" 
Published in Proceedings of Journal of Machine Learning Research 2017. 

Step by step Guide.
============
1. Preliminaries: g++, linux.
------------
2. Running the code

-- cd <the path of the code file>

-- make

-- ./ssh <data set file> <query time series file> <time series length> <number of time series> <filter length> <shift size> <shingle length> <local constraint of dtw>

3. Running example for the given dataset

-- cd ~/SSH

-- make

-- ./ssh data query 1024 2000 100 3 15 10

If you have any qiestions or problems regarding the code, please feel free to contact Chen Luo (cl67@rice.edu)
