

class SortedQueue(list):


    def __init__(self,initlist, maxsize=None,reverse_sort=False,cmp=None,key=None):
        super(SortedQueue, self).__init__(initlist)
        self.cmp = cmp
        self.key = key
        self.reverse_sort = reverse_sort
        self.maxsize = maxsize



    def __add__(self, other):
        super(SortedQueue, self).__add__(other)
        self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)

    def __iadd__(self, other):
        # print self
        super(SortedQueue, self).__iadd__(other)
        # print self
        self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)
        # print self
        return self


    def tolist(self):
        return list(self)

    def append(self, p_object):
        super(SortedQueue, self).append(p_object)
        # print self
        self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)
        # print self

    def sort(self, cmp=None, key=None, reverse=False):
        # print self
        super(SortedQueue, self).sort(cmp=cmp,key=key,reverse=reverse)
        self.__trim__()


    def insert(self, index, p_object):
        super(SortedQueue, self).insert(index,p_object)
        self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)

    @property
    def reverse_sort(self):
        return self._reverse_sort

    @reverse_sort.setter
    def reverse_sort(self,rev):
        if(not isinstance(rev,bool)):
            raise TypeError('reverse_sort must be a bool type!')
        else:
            self._reverse_sort = rev
            self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)



    @property
    def maxsize(self):
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize):
        if(not isinstance(maxsize, (int,long)) and maxsize is not None):
            raise TypeError('maxsize must be an int or long!')
        else:
            self._maxsize = maxsize
            self.sort(reverse=self.reverse_sort,cmp=self.cmp,key=self.key)

    def __trim__(self):
        if(self.maxsize is not None):
            if(len(self) > self.maxsize):
                for i in range(len(self) - self.maxsize):
                    self.pop()
