import numpy as np

class TetSearch(object):
    def __init__(self, tet_list) -> None:
        super().__init__()
        #List of tetrahedra, each row contains the index of 4 vertices defining the tetrahedron
        self.tet_list = tet_list
        #To make finding faces faster, we sort the tet by the first, second, third and fourth column
        # and store each sort inside a row of tet_sort. This sorting enables efficient face retrieval.
        # Nth row stores tet stored by their Nth column
        self.col_sorted_tetlist = []
        #Nth row stores the indices of the sort according to the Nth column
        # such that if sorted_to_unsorted[0][5]==10 then the 5th tet when sorting according to the 0th column is
        # the 10th tet in the larger, unsorted list. It allows to go from the sorted domain to the unsorted.
        self.sorted_to_unsorted = []
        #The Mth column of the Nth row of start_at_idx gives the index where the Nth row of tet_sort starts to
        # have tets containing vertex M. Example: If start_at_idx[1][42]==100 and start_at_idx[1][43]==103, you
        # will find vertex #42 in col_sorted_tetlist[1][100], col_sorted_tetlist[1][101] and col_sorted_tetlist[1][102].
        self.start_at_idx = []

        #Builds the lists
        self.setup()

    def setup(self):
        #Iterate of the the four columns of the tet_list
        for c in range(0,4):
            #Sort by the c-th column
            ind = self.tet_list[:, c].argsort()
            tetras_trimmed_sorted = self.tet_list[ind]
            col = tetras_trimmed_sorted[:,c]
            #The indices of the rows that have X as their c-th element are
            # start_idx[X] (included) to start_idx[X+1] (excluded).
            start_idx = [0]
            for i in range(0,self.tet_list.max()):
                for j in range(start_idx[-1], len(self.tet_list)):
                    if col[j] > i:
                        start_idx.append(j)
                        break
            for i in range(len(start_idx), self.tet_list.max()+1):
                start_idx.append(j)
            self.start_at_idx.append(start_idx)
            self.col_sorted_tetlist.append(tetras_trimmed_sorted)
            self.sorted_to_unsorted.append(ind)

    #Search for a tet that shares face f with tet t by iterating only over
    # tet that share at least one vertex.
    def find_face_neighbor(self, f, t):
        if not self.start_at_idx:
            self.setup()
        #For each column of the tet
        for c in range(0,4):
            #For each vertex of the face
            for i in range(0,3):
                vertex = f[i]
                #Find the range of tet that share this vertex (promixing range)
                start_ind = self.start_at_idx[c][vertex]
                if vertex == len(self.start_at_idx[c])-1:
                    end_ind = start_ind
                else:
                    end_ind   = self.start_at_idx[c][vertex+1]
                #For each tet in the promising range
                for j in range(start_ind, end_ind):
                    if len(set(self.col_sorted_tetlist[c][j]) & set(f)) == 3 and \
                        not np.array_equal(self.col_sorted_tetlist[c][j],t):
                        return self.col_sorted_tetlist[c][j], [self.sorted_to_unsorted[c][j].tolist()]
        return None, None