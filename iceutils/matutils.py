#-*- coding: utf-8 -*-

# Some fast matrix multiply routines from GIAnT
def dmultl(dvec, mat):
    """
    Left multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res    -> dot (diag(dvec), mat)
    """

    res = (dvec*mat.T).T
    return res

def dmultr(mat, dvec):
    """
    Right multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res     -> dot(mat, diag(dvec))
    """

    res = dvec*mat
    return res

# end of file
