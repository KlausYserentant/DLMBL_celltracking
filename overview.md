# Celltracking in 3D

* Linear assignment: Link cells in frames t and t+1

* In exercise: Euclidian distance

* Our approach

  * Train a model to build embedding space where cell A in frame t and t+1 

    

# Tasks

## 0 - Denoising

* Skip for now

## 0 - Centroids from unannotated data

* StarDist 
* Skip for now

## 1 - Data Generator

* Load pre-defined images (4D)
  * Raw data format?
* Load centroid positions
  * Raw data format?
* Load GT trajectories
  * Raw data format?
* Define data class
  * image class
  * trajectory class
* Splitting Training/Test
* Volume pair formation
* Augmentation
* Balancing
* Batches

## 2 - Model

* VGG ?!
  * Conv3d
* Loss: Cosine
* Depth?
* Volume size?
* Length output vector?
* Embedding



# 3 - LAP + connect tracks

* Linear assignment
* Make use of Euclidean distance and Embedding vectors
* Introduce weighting to balance between Euclidean distance and embedding vectors
  * d([x~1~,y~1~],[x~2~,y~2~]) + cd ([a~1~,b~1~],[a~2~,b~2~]
  * [x,y,a,b]
  * 0 < c < inf




