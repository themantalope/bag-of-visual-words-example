import numpy as np
import .node
from .sortedqueue import SortedQueue


class KDTree(object):
    """
    A KDTree object. Initialize it with either a list of lists, a numpy.ndarray, or a numpy.matrix (will be converted
    to a numpy.ndarray). The default behavior is to assume that for a data matrix of m x n, where n < m, the number of
    dimensions in the KD-Tree is n and the number of datapoints to organize is m. This behavior can be overridden
    by setting the data_axis variable upon initialization. When the object is initialized, the tree is automatically
    created.
    """
    def __init__(self,data,data_axis=None,id_list=None):
        #first figure out what the data type is
        """

        :rtype : object
        """
        if(isinstance(data,np.matrix)):
            data = np.array(data)
        elif(isinstance(data, list)):
            #check to see that it's a list of lists
            allLists = True
            allSameLen = True
            listLen = 0
            if(isinstance(data[0],list)):
                listLen = len(data[0])
            else:
                raise TypeError('Data must be a list of lists with the same length in each sublist!')

            if(listLen == 0):
                raise TypeError('Data must be a list of lists with the same length in each sublist!')


            for element in data:
                if(not isinstance(element, list)):
                    allLists = False
                    break
                elif(len(element) != listLen):
                    allSameLen = False
                    break


            if(not allLists or not allSameLen):
                raise TypeError('Data must be a list of lists with the same length in each sublist!')

            #ok if we made it here, should be ok
            data = np.array(data)
            # print data
        elif(not isinstance(data,np.ndarray)):
            #if it's not the other things and not an nd-array
            raise TypeError('Data must be a numpy.ndarray, numpy.matrix or a list of lists!')

        #made it here, check which dimension to split the data
        if(data_axis is not None):
            if(data_axis == 1):
                data = data.T

            self.nDims = data.shape[1]

        else:
            rows, cols = data.shape
            if(rows < cols):
                self.nDims = rows
                data = data.T
            else:
                self.nDims = cols

        #this way, the rows are always the points to organize and the columns are the data per dimension
        #if we got an id list, check to see that the len of the id list is the same as the number of samples
        if(id_list is not None):
            nSamples = data.shape[0]
            if(not isinstance(id_list,list) and not isinstance(id_list,np.ndarray)):
                raise TypeError('id_list type is not recognized. Must be either a list or a numpy.ndarray.')
            else:
                if(nSamples != len(id_list)):
                    raise ValueError('The number of ids in the id list must be the same as the number of data points!')

            id_list = np.array(id_list)

        #well if you made it this far, it's time to start building the tree

        self._buildTree(data,data.shape[1],id_list=id_list)




    def _buildTree(self,data, nDim, id_list = None):
        #move along the axes, getting the median axes per cycle, then specify the right and left branches
        #first find the root
        splitDim = depth = 0
        curCol = data[:,splitDim]
        # print 'first col: ', curCol

        median_row_index = np.abs(curCol - np.median(curCol)).argmin()

        left_row_idxs = np.where(curCol < data[median_row_index,splitDim])
        left_row_idxs = left_row_idxs[0]
        right_row_idxs = np.where(curCol > data[median_row_index,splitDim])
        right_row_idxs = right_row_idxs[0]
        left_data = data[left_row_idxs,:]
        right_data = data[right_row_idxs,:]

        # print 'First median data: ', data[median_row_index,:]
        # print 'First left data:\n ', left_data
        # print 'First right data:\n ', right_data
        self.root = node.Node(data[median_row_index,:],splitDim,depth)

        if(id_list is not None):
            # print id_list
            self.root.ID = id_list[median_row_index]
            self.root.right_child = self._buildTreeRecursive(right_data, depth + 1, nDim, id_list = id_list[right_row_idxs])
            self.root.left_child = self._buildTreeRecursive(left_data, depth + 1, nDim, id_list = id_list[left_row_idxs])
        else:
            self.root.right_child = self._buildTreeRecursive(right_data, depth + 1, nDim)
            self.root.left_child = self._buildTreeRecursive(left_data, depth + 1, nDim)



    def _buildTreeRecursive(self, data, depth, nDim, id_list= None):
        #first check to see if we are done
        if(data.shape[0] < 1):
            return
        else:
            splitDim = depth % nDim
            curCol = data[:,splitDim]
            # print 'Current data: ', curCol
            #find the mid point of that dimension
            # print 'Median: ', np.median(curCol)
            median_row_index = np.abs(curCol - np.median(curCol)).argmin()
            # print 'Median row: ', median_row_index
            # median_row_index = median_row_index[0]
            # print 'Median row: ', median_row_index


            # print 'Print median data point: ', curCol[median_row_index]

            left_row_idxs = np.where(curCol < curCol[median_row_index])
            left_row_idxs = left_row_idxs[0]
            right_row_idxs = np.where(curCol < curCol[median_row_index])
            right_row_idxs = right_row_idxs[0]
            left_data = data[left_row_idxs,:]
            right_data = data[right_row_idxs,:]

            newNode = node.Node(data[median_row_index,:],splitDim,depth)
            if(id_list is not None):
                newNode.ID = id_list[median_row_index]
                newNode.right_child = self._buildTreeRecursive(right_data, depth + 1, nDim, id_list=id_list[right_row_idxs])
                newNode.left_child = self._buildTreeRecursive(left_data, depth + 1, nDim, id_list=id_list[left_row_idxs])
            else:
                newNode.right_child = self._buildTreeRecursive(right_data, depth + 1, nDim)
                newNode.left_child = self._buildTreeRecursive(left_data, depth + 1, nDim)


            return newNode


    def findkNN(self, point, k=1):
        nn_list = SortedQueue([], maxsize=k, key=lambda obj: obj[0])
        # start at the root, navigate to the leaves
        first_node = self.root
        dist = self
