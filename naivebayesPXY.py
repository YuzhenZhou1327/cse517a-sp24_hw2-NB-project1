#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo (smoothing with all zeros)
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#
#    Output:
#    posprob: dx1 probability vector with entries p(x_alpha = 1|y=+1)
#    negprob: dx1 probability vector with entries p(x_alpha = 1|y=-1)
# =============================================================================



    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    # TODO: do not use np.matrix!
    X = np.matrix(x)
    Y = np.matrix(y)

    d,n = X.shape

    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.array([[-1, 1]])

    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
    Ynew = np.hstack((Y, Y0))


    # matrix of all-zeros -
    X1 = np.zeros((d, 2))
    # add one all-zeros positive and negative example - M.Joo
    Xnew = np.hstack((Xnew, X1))
    Ynew = np.hstack((Ynew, Y0))

    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    

# =============================================================================
# fill in code here
    # YOUR CODE HERE  # Flatten Ynew to use as a mask
    x_pos = np.sum(Xnew[:,np.asarray(Ynew == 1).flatten()],axis=1)
    x_neg = np.sum(Xnew[:,np.asarray(Ynew == -1).flatten()],axis=1)
    count = 0
    sum = np.sum(Xnew, axis = 0)
    count = np.sum(Ynew == 1)
    posprob = x_pos / count

    count = np.sum(Ynew == -1)
    negprob = x_neg / count
    return posprob, negprob 
#================================================
