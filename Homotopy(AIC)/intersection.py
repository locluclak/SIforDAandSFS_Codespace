import numpy as np
from bisect import bisect

class Obj:
    def __init__(self, v, bracket: bool):
        """
        False: ( 
        True: )
        """
        self.value = v
        self.ooc = bracket
    def __lt__(self, other):
        if self.value < other.value:
            return True
        return False

    def show(self):
        if self.ooc == False:
            print(f"({self.value}", end=" ")
        else:
            print(f"{self.value})", end=" ")


def solvequadra(a, b, c):
    """ ax^2 + bx +c < 0 """
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return ((-np.inf, np.inf),)
        else:
            print("Error to find interval. ")
    # print("delta:", delta)
    x1 = (- b - np.sqrt(delta)) / float(2*a)
    x2 = (- b + np.sqrt(delta)) / float(2*a)
    # if x1 > x2:
    #     x1, x2 = x2, x1  
    if a < 0:
        return ((-np.inf, x2),(x1, np.inf))
    return ((x1,x2),)

def intersec_local(Raxis: list, interval: tuple):
    """
    Intersection for intervals from quadratic inequation 
    """
    ls = Raxis.copy()

    a, b = Obj(interval[0][0], False), Obj(interval[0][1], True)
    indexO = bisect(ls, a)
    indexC = bisect(ls, b)

    if indexO == len(ls):
        return []
    else:
        temp = []
        for i in range(indexO, min(indexC, len(ls))):
            temp.append(ls[i])
        if ls[indexO - 1].ooc == False:
            temp = [a] + temp
        if temp[-1].ooc == False:
            temp.append(b)
        ls1 = temp
    
    if len(interval) == 1:
        return ls1
    else:
        ls = ls1 + ls[indexC:].copy()
        
        c, d = Obj(interval[1][0], False), Obj(interval[1][1], True)
        indexO = bisect(ls, c)
        indexC = bisect(ls, d)

        if indexO == len(ls):
            return ls1
        else:
            temp = []
            for i in range(indexO, min(indexC, len(ls))):
                temp.append(ls[i])

            if ls[indexO - 1].ooc == False or temp[0].ooc == True:
                temp = [c] + temp
            if temp[-1].ooc == False:
                temp.append(d)
            ls2 = temp

            return ls1 + ls2 
    # raise "Error return"

def Intersection(a: list) -> list:
    Raxis = [Obj(-np.inf,False), Obj(np.inf,True)]
    for each in a:
        Raxis = intersec_local(Raxis, each)

    res = []
    for i in range(0, len(Raxis), 2):
        res.append((Raxis[i].value, Raxis[i+1].value))
    return res
def Intersec_quad_linear(a: list, e: tuple) -> list:
    Raxis = []
    for i in range(0, len(a)):
        Raxis.append(Obj(a[i][0], False))
        Raxis.append(Obj(a[i][1], True ))
    # print(e)
    Raxis = intersec_local(Raxis, e)

    res = []
    for i in range(0, len(Raxis), 2):
        res.append((Raxis[i].value, Raxis[i+1].value))
    return res
if __name__ == "__main__":
    interval1 = ((-98369550.69527487, -0.3366092581332503),)
    interval2 = ((0.3366092581332504, 98369544.42982495),)
    # interval3 = ((-1e-26, 1e-26),)
    interval4 = ((-0.5714060545812975, -0.3056044608492629),)

    interval_list = [interval1, interval2, interval4]
    # Raxis = [Obj(-np.inf,False), Obj(np.inf,True)]
    
    # for each in interval_list:
    #     Raxis = intersec_local(Raxis, each)

    # for x in Raxis:
    #     x.show()

    print(Intersection(interval_list))