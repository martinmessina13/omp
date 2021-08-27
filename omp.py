import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from qpsolvers import solve_ls

def OMP_C(y_r, M_r, M_mp, M_mn, D, K_omp, epsilon_omp, CLIP_LEVEL, /, mode, verbose):
    '''
    OMP_C
    Orthogonal Matching Pursuit (OMP) declipping algorithm.
    Returns the sparsest representation of a given vector, provided a DCT dictionary.
    
    Parameters
    y_r: The reliable samples of the audio signal.
    M_r: The measurement matrix such that y_r = M_r · s, where s is the reconstructed audio signal.
    D: The DCT dictionary, a matrix comprised by elementary signals called atoms.
    K_omp: The number of discrete frequency bins.
    epsilon_omp: The tolerance threshold.
    CLIP_LEVEL: The maximum absolute value of the clipped audio signal.
    mode: Whether to enforce the unconstrained ('normal') or the minimum constrained ('min') problem.
    '''
    
    # Initialization of data structures.
    K = D.shape[1]
    K_support = np.arange(K)
    
    # Normalization matrix.
    W = np.identity(K)
    W = W * 1/np.linalg.norm(M_r.dot(D), axis=0)
    
    # The dictionary is multiplied with measurement matrix M_r and its columns normalized.
    D_ = M_r.dot(D).dot(W)
    
    # Frequency counter.
    k = 0
    
    # Support set.
    support = set()
    
    # Residual vector
    r = y_r
    
    # Sparse vector memory allocation.
    x_k = np.zeros(K)
    
    # The length of the missing samples vectors is stored for later use.
    L_Imp = M_mp.shape[0]
    L_Imn = M_mn.shape[0]
    
    # The loop stops when the error threshold is reached or when all columns of the dictionary are visited.
    while k < K_omp and r.dot(r) > epsilon_omp:
        
        # The frequency counter k is increased by one.
        k += 1
        
        # The maximum correlation value will be stored in a variable that is initialized to zero value.
        max_corr = 0
        
        # The residual is reshaped as a multidimensional array to take advantage of numpy's matrix multiplication capabilities.
        r = r.reshape(-1, 1)
        
         # We compute the dot product between all the atoms and the current residual. This gives us the correlation between the vectors.
        correlation = np.abs(np.sum(r * D_, axis=0))
        
        # The atom index related to the highest correlation is saved in memory.
        atom = np.argmax(correlation)
        
        # The atom index is stored inside a set.
        support.add(atom)
        
        # Then, the support set is casted into a list for later use.
        support_list = list(support)
        
        # The length of the support and the reliable samples vector is obtained.
        L_sup = len(support)
        L_r = len(y_r)
        
        # Here we perform the memory allocation of a new dictionary built from the chosen atoms.
        D__ = np.empty((L_r, L_sup))
        
        # The chosen atoms are assigned to the empty dictionary.
        D__[:, :L_sup] = D_[:, support_list]
        
        # Now it's time to obtain the sparse vector. As we are dealing with an underdetermined system (infinite solutions), we will obtain the solution with the minimum orthogonal error by means of a least-squares projection.
        # The projection is computed using the pseudoinverse. We will use the linalg numpy module for this task.
        x_k[support_list] = np.linalg.pinv(D__).dot(y_r)
        
        # And we update the residual using the previous result.
        r = y_r - D__.dot(x_k[support_list])
        
        if verbose:
            if k % 50 == 0 and not (k == K_omp or r.dot(r) < epsilon_omp):
                print('iteration n º {}, error: {:.4f}, support set elements: {}'.format(k, r.dot(r), L_sup)) 
            elif k == K_omp or r.dot(r) < epsilon_omp:
                print('total number of iterations: {}, error: {:.4f}, support set elements: {}'.format(k, r.dot(r), L_sup))

        # A final update on x_k is made in order to improve the solution for declipping if any of the missing samples vectors are not zero.
        # The corresponding update will be based on the current value of the mode argument.
        if mode == 'normal':
            pass
        elif mode == 'min':
            MAX_LEVEL = np.inf
            
            # The following conditional checks if the maximum sparsity level or the minimum residual error has been reached. 
            # It also considers if there are positive, negative or positive and negative clipped samples in the current frame.
            if (k == K_omp or r.dot(r) < epsilon_omp) and (L_Imp != 0 or L_Imn != 0):
                if verbose: print('Entering optimizer...')
                
                if L_Imp != 0 and L_Imn != 0:
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1, G2, G3))

                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1, h2, h3))

                elif L_Imp != 0: 
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1))
                    
                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1))
                    
                elif L_Imn != 0: 
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    
                    G = np.vstack((G2, G3))
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h2, h3))
                    

                # We implement a convex optimization solver for the constrained least squares (CLS) problem.
                x_k[support_list] = solve_ls(D__, y_r, G, h, solver='cvxpy', verbose=False)

                # If there is no solution to the convex optimization problem, use the unconstrained solution.
                if np.isnan(x_k[support_list][0]):
                    x_k[support_list] = np.linalg.pinv(D__).dot(y_r)

                if verbose: print('...exiting optimizer')      
            
    # Finally, we return the normalized sparse representation vector.
    return W.dot(x_k)

def OMP_G(y_r, I_r, M_r, M_mp, M_mn, D, K_omp, epsilon_omp, CLIP_LEVEL, /, mode, verbose):
    '''
    OMP_G
    Orthogonal Matching Pursuit (OMP) declipping algorithm.
    Returns the sparsest representation of a given vector, provided a Gabor dictionary.
    
    Parameters
    y_r: The reliable samples of the audio signal.
    M_r: The measurement matrix such that y_r = M_r · s, where s is the reconstructed audio signal.
    D: The Gabor dictionary, a matrix comprised by elementary signals called atoms.
    K_omp: The number of discrete frequency bins.
    epsilon_omp: The tolerance threshold.
    CLIP_LEVEL: The maximum absolute value of the clipped audio signal.
    mode: Whether to enforce the unconstrained ('normal') or the minimum constrained ('min').
    '''
    
    K = D.shape[1] // 2
    K_support = np.arange(2 * K)

    W = np.identity(2 * K)
    W = W * 1/np.linalg.norm(M_r.dot(D), axis=0)
            
    D_ = M_r.dot(D).dot(W)
    
    # DCT and DST dictionaries. Their concatenation results in a dictionary that considers the phase component of audio signals.
    D_c_, D_s_ = D_[:, :K], D_[:, K:]
    
    k = 0
    
    support = set()
    
    r = y_r
    
    x_k = np.zeros(2 * K)
    
    L_Imp, L_Imn = M_mp.shape[0], M_mn.shape[0]
    
    while k < K_omp and r.dot(r) > epsilon_omp:
        
        k += 1
        
        r = r.reshape(-1, 1)
        
        # Gabor's dictionary atom selection step.
        x_c_ = np.array([(np.sum((D_c_ * r), axis=0) - np.sum(D_c_ * D_s_, axis=0) * np.sum(D_s_ * r, axis=0))/(1 - np.sum(D_c_ * D_s_, axis=0) ** 2)])
        x_s_ = np.array([(np.sum(D_s_ * r, axis=0) - np.sum(D_c_ * D_s_, axis=0) * np.sum(D_c_ * r, axis=0))/(1 - np.sum(D_c_ * D_s_, axis=0) ** 2)])
        
        # Objective function that we want to minimize.
        result = np.linalg.norm(r - D_c_ * x_c_ - D_s_ * x_s_, axis=0) ** 2
        
        atom = np.argmin(result)
        
        # In this case, the atom index of the corresponding sine atom is also stored.
        support.add(atom)
        support.add(atom + K)
        
        support_list = list(support)
        
        L_sup = len(support)
        L_r = len(y_r)
        
        D__ = np.empty((L_r, L_sup))
        
        D__[:, :L_sup] = D_[:, support_list]
        
        x_k[support_list] = np.linalg.pinv(D__).dot(y_r)

        r = y_r - D__.dot(x_k[support_list])
        
        if verbose:
            if k % 50 == 0 and not (k == K_omp or r.dot(r) < epsilon_omp):
                print('iteration n º {}, error: {:.4f}, support set elements: {}'.format(k, r.dot(r), L_sup)) 
            elif k == K_omp or r.dot(r) < epsilon_omp:
                print('total number of iterations: {}, error: {:.4f}, support set elements: {}'.format(k, r.dot(r), L_sup))

        if mode == 'normal':
            pass
        elif mode == 'min':
            MAX_LEVEL = np.inf
            
            if (k == K_omp or r.dot(r) < epsilon_omp) and (L_Imp != 0 or L_Imn != 0):
                if verbose: print('Entering optimizer...')
                
                if L_Imp != 0 and L_Imn != 0:
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1, G2, G3))

                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1, h2, h3))

                elif L_Imp != 0: 
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1))
                    
                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1))
                    
                elif L_Imn != 0: 
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    
                    G = np.vstack((G2, G3))
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h2, h3))
                    
                x_k[support_list] = solve_ls(D__, y_r, G, h, solver='cvxpy', verbose=False)

                if np.isnan(x_k[support_list][0]):
                    x_k[support_list] = np.linalg.pinv(D__).dot(y_r)

                if verbose: print('...exiting optimizer')      
            
    return W.dot(x_k)
