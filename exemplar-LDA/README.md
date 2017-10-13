# Generic version of WHO for object detection with LDA models


## COPYRIGHT

This package contains Matlab source code for object detection with LDA models.

This repo is based on the original version of who from 
(Bharath Hariharan, Jitendra Malik and Deva Ramanan. 
Discriminative decorrelation for clustering and classification. In ECCV 2012 ).
We significantly altered it, making it more flexible, easier to access, 
and less focused on hog features in general.

## START / SETUP

Compile mex-files via  
```
compileWHOEfficient
```

Setup workspace  
```
initWorkspaceWHOGeneric
```

## RUN E-LDA FOR OUTLIER DRTECTION
```
exemplar_LDA
```
