import numpy as np
from bisect import bisect

class Obj:
    def __init__(self, v, bracket: str):
        self.value = v
        self.ooc = bracket
    def __lt__(self, other):
        if self.value < other.value:
            return True
        return False

    def show(self):
        if self.ooc == "(":
            print(f"{self.ooc}{self.value}", end=" ")
        else:
            print(f"{self.value}{self.ooc}",end=" ")

ls = [Obj(-np.inf, "("),  Obj(np.inf, ")")]
# ls = []
interval0 = ((-2,3),)
a, b = Obj(interval0[0][0], "("), Obj(interval0[0][1], ")")

indexO = bisect(ls, a)
indexC = bisect(ls, b)

print(indexO, indexC)

if indexO == len(ls):
    ls = []
else:
    temp = []
    for i in range(indexO, min(indexC, len(ls))):
        temp.append(ls[i])
    if ls[indexO - 1].ooc == "(":
        temp = [a] + temp
    if temp[-1].ooc == "(":
        temp.append(b)
    ls = temp
for x in ls:
    x.show() 