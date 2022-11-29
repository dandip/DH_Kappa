# DH_Kappa
Numpy implementation of the DiPietro-Hazari Kappa.

## What is the DiPietro-Hazari Kappa?

It's a new statistical metric for assessing labels in the context of inter-annotator agreement. Read more about the theoretical underpinnings here: https://arxiv.org/pdf/2209.08243.pdf

## Using this Code

To compute the DH Kappa, use the function `dh_kappa(A,B)`:
```
Computes DH Kappa for annotation matrix A and label matrix B
Suppose you have n pieces of data d_1, ... , d_n with m possible categories
c_1, ... , c_m. Each piece of data is assessed by N annotators; there are n
proposed labels as well, denoted l_1, ... , l_n, where each l_i denotes the
proposed category label of d_i. Let A define the n x m matrix where a_ij
indicates the number of annotators that placed d_i in c_j. Let B denote the
n x m matrix where b_ij = 1 if l_i = c_j and 0 otherwise. Using these
matrices A and B, this function computes the DiPietro-Hazari Kappa metric.

~ Parameters ~
- A (n x m) numpy matrix
- B (n x m) numpy matrix

~ Returns (float) ~ : DH Kappa value
```
