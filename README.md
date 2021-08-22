# Orthogonal Matching Pursuit

Python implementation of the Orthogonal Matching Pursuit declipping algorithm by Amir Adler, Valentin Emiya, Maria Jafari, Michael Elad, Rémi Gribonval, et al. (“A constrained matching pursuit approach to audio declipping,” in
2011 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), May 2011, pp. 329–332.).

The algorithm uses the sparse representation (SR) framework along with DCT (Discrete Cosine Transform) or DGT (Discrete Gabor Transform) overcomplete dictionaries and solves a constrained least squares optimization problem to estimate clipped, degraded or missing samples in audio signals.
