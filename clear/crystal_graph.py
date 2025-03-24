"""
    Module name: cgml.crystal_graph
    
    This module contains the CrystalGraph class that
    generates a crystal graph from the given structure
    in the form of a dictionary
"""


class CrystalGraph:
    """
    Based on NN values, the crystal graph is built at each atomic site
    and contains a method to create a graph
    """

    def __init__(self, structure, nn_algorithm):
        self.structure = structure
        self.nn_algorithm = nn_algorithm
        self.graph = self.__get_graph()

    def __get_graph(self):
        """
        Function to create a graph dictionary created
        on the basis of the atoms at each site  
        """
        graph_dict = {}
        sites = self.structure.sites
        for i in range(len(sites)):
            site_list = []
            for item in self.nn_algorithm.get_nn_info(self.structure, i):
                site_list.append(item['site_index'])
            site_list = sorted(site_list)
            graph_dict[i] = site_list
        return graph_dict
