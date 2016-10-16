

cdef cMultiply(double a, double b):
    return a * b

cpdef cpSum(double a, double b):
    return a + b

def pyTestMultiply():
    return cMultiply(2,2)

def pyTestSum():
    return cpSum(2,2)
