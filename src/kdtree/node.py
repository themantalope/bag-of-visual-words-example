

class Node(object):

    def __init__(self,dimension,depth,right_child=None,left_child=None,ID = None):
        self.isLeaf = True
        self.dimension = dimension
        self.depth = depth
        self.right_child = right_child
        self.left_child = left_child
        self.ID = ID



    @property
    def left_child(self):
        return self.__left_child

    @left_child.setter
    def left_child(self,left_child):
        """

        :type left_child: Node
        """
        if(not isinstance(left_child, Node)):
            raise TypeError('The child must be a branch or leaf node object!')
        else:
            self.__left_child = left_child


    @property
    def right_child(self):
        return self.__right_child

    @right_child.setter
    def right_child(self,right_child):
        """

        :type right_child: Node
        """
        if(not isinstance(right_child, Node)):
            raise TypeError('The child must be a branch or leaf node object!')
        else:
            self.__right_child = right_child





class LeafNode(Node):
    """
    This is a class that encodes a node within a kd-tree. Will have a right and left neighbor, which are also nodes.
    The node object also keeps track of which dimension data structure it is in.
    """

    def __init__(self, vector, dimension, depth,right_child=None,left_child=None,ID = None):
        super(LeafNode, self).__init__(dimension, depth,right_child,left_child,ID)
        self.vector = vector






class BranchNode(Node):

    def __init__(self, splitting_value, dimension, depth,right_child=None,left_child=None,ID = None):
        super(BranchNode, self).__init__(dimension,depth,right_child,left_child,ID)
        self.splitting_value = splitting_value
