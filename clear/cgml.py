"""
    Module name: cgml.cgml
    
    This module contains the CGML class that helps to
    generate a data sheet containing features that
    can be utilised for machine learning
"""

import os

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.vasp.inputs import Poscar

from cgml.crystal_graph import CrystalGraph

_PROPERTY_LIST = [
    'Density', 'AtomicNumber', 'MendeleevNumber', 'AtomicWeight', 'BoilingT',
    'MeltingT', 'EA', 'Group', 'Period', 'Radius', 'EN', 'NsValence',
    'NpValence', 'NdValence', 'NfValence', 'NValence', 'NsUnfilled',
    'NpUnfilled', 'NdUnfilled', 'NUnfilled', 'GSvolume_pa', 'GSmagmom',
    'SpaceGroup', 'fracSvalelec', 'fracPvalelec', 'fracDvalelec',
    'fracFvalelec', 'VEC'
]


class CGML:
    """
    Class that implements methods to run various operations
    on the input data sheet, analyse structures of HEAs from the 
    POSCAR files available and create a data sheet containing
    values of various properties that can be used as features
    for machine learning applications
    """

    def __init__(self, poscar_directory_path, data_sheet_path):
        self.poscar_directory_path = poscar_directory_path
        self.data_sheet_path = data_sheet_path
        self.input_df = pd.read_excel(self.data_sheet_path)
        self.export_path = 'cg_features.csv'
        self.final_df = pd.DataFrame()
        self.prop_dict = {}
        self.prop_list = _PROPERTY_LIST
        self.list_of_poscars = []

    def set_export_path(self, export_path):
        """
        Function to set the custom export path
        ...
        Parameters
        ----------
        export_path : str
            The export path given by the user
        """

        self.export_path = export_path

    def __get_distance(self, positions, graph):
        """
        Function to compute and store atomic distances less than 3 ang 
        for all NN with its target atom
        ...
        Parameters
        ----------
        positions : numpy.ndarray
            Two dimensional array containing the positions of the atoms in the crystal
        graph : dict
            Dictionary containing the graph representation of the crystal
        """

        distance = {}
        for i in range(len(graph)):
            dist = []
            for j in graph[i]:
                calc_dist = np.sqrt((positions[j][2] - positions[i][2])**2 +
                                    (positions[j][1] - positions[i][1])**2 +
                                    (positions[j][0] - positions[i][0])**2)
                dist.append(calc_dist)
            distance[i] = dist

        return distance

    def __get_composition(self, poscar_path):
        """
        Function to read poscar file and get the composition of each element
        ...
        Parameters
        ----------
        poscar_path : str
            Path to the poscar file to be read
        """

        all_lines = []
        with open(poscar_path, 'r', encoding='utf-8') as read_file:
            all_lines = read_file.readlines()

        name_line = all_lines[5].split()
        comp_line = all_lines[6].split()
        comp = []
        s_line = 0

        for _, val in enumerate(comp_line):
            s_line = s_line + float(val)

        for _, val in enumerate(comp_line):
            c_calc = float(val) / s_line
            comp.append(c_calc)
            lists = [name_line, comp]
            df_comp = pd.DataFrame(lists)
            df_comp = df_comp.transpose()
            df_comp.columns = ["name", "comp"]

        return df_comp

    def __get_elemental_property(self, read_poscar, graph, positions, prop,
                                 comp, comp_rows, list_of_elems, rows):
        """
        Function to calculate each individual feature from the input datasheet
        ...
        Parameters
        ----------
        read_poscar: ase.atoms.Atoms
            The POSCAR file that has been read
        graph: dict
            Dictionary containing the graph representation of the crystal
        positions : numpy.ndarray
            Two dimensional array containing the positions of the atoms in the crystal
        prop: str
            The property name for which features are computed
        comp: pd.DataFrame
            The composition of the HEA read from the POSCAR file
        comp_rows: pandas.core.series.Series
            The row values of the comp dataframe
        list_of_elems: str []
            List containing element names
        rows: pandas.core.series.Series
            The row values of the input data sheet
        """

        final = []
        for i in graph:
            x_1 = [atom.symbol for atom in read_poscar if atom.index == i]
            x_1 = ''.join(x_1)
            weight_sum = 0
            dist_sum = 0

            for j in graph[i]:
                x_2 = [atom.symbol for atom in read_poscar if atom.index == j]
                x_2 = ''.join(x_2)
                dist = np.sqrt((positions[j][2] - positions[i][2])**2 +
                               (positions[j][1] - positions[i][1])**2 +
                               (positions[j][0] - positions[i][0])**2)

                for cdx, c_row in enumerate(comp_rows):
                    if x_1 == c_row:
                        c1_idx = cdx
                    if x_2 == c_row:
                        c2_idx = cdx
                        comp_1 = comp['comp'].values[c1_idx]
                        comp_2 = comp['comp'].values[c2_idx]

                if x_1 and x_2 in list_of_elems:
                    for idx, val in enumerate(rows):
                        if x_1 == val:
                            x_1_idx = idx
                        if x_2 == val:
                            x_2_idx = idx
                            weight1 = np.absolute(
                                (comp_1 *
                                 (self.input_df[prop].values[x_1_idx])) -
                                (comp_2 *
                                 (self.input_df[prop].values[x_2_idx])))
                            weight = dist * weight1
                            weight_sum = weight_sum + weight
                            dist_sum = dist_sum + dist
                else:
                    print('Element not in list')
            fin = weight_sum / dist_sum
            final.append(fin)

        self.prop_dict[prop] = final

    def __create_feature_sheet(self, name):
        """
        Function to create the datasheet containing the required features
        ...
        Parameters
        ----------
        name : str
            The name of the POSCAR for which the features are being calculated
        """
        val_list = []
        for item in self.prop_dict.items():
            val_list.append(np.mean(list(item)[1]))
            val_list.append(np.std(list(item[1])))
            val_list.append(np.max(list(item[1])))
            val_list.append(np.min(list(item[1])))

        created_df = pd.DataFrame(
            val_list,
            index=[
                "D_mean", "D_std", "D_max", "D_min", "AN_mean", "AN_std",
                "AN_max", "AN_min", "MN_mean", "MN_std", "MN_max", "MN_min",
                "AW_mean", "AW_std", "AW_max", "AW_min", "TB_mean", "TB_std",
                "TB_max", "TB_min", "TM_mean", "TM_std", "TM_max", "TM_min",
                "EA_mean", "EA_std", "EA_max", "EA_min", "Group_mean",
                "Group_std", "Group_max", "Group_min", "Period_mean",
                "Period_std", "Period_max", "Period_min", "R_mean", "R_std",
                "R_max", "R_min", "EN_mean", "EN_std", "EN_max", "EN_min",
                "Sval_mean", "Sval_std", "Sval_max", "Sval_min", "Pval_mean",
                "Pval_std", "Pval_max", "Pval_min", "Dval_mean", "Dval_std",
                "Dval_max", "Dval_min", "Fval_mean", "Fval_std", "Fval_max",
                "Fval_min", "Nval_mean", "Nval_std", "Nval_max", "Nval_min",
                "Sunfil_mean", "Sunfil_std", "Sunfil_max", "Sunfil_min",
                "Punfil_mean", "Punfil_std", "Punfil_max", "Punfil_min",
                "Dunfil_mean", "Dunfil_std", "Dunfil_max", "Dunfil_min",
                "Nunfil_mean", "Nunfil_std", "Nunfil_max", "Nunfil_min",
                "Vol_mean", "Vol_std", "Vol_max", "Vol_min", "mag_mean",
                "mag_std", "mag_max", "mag_min", "Spacegrp_mean",
                "Spacegrp_std", "Spacegrp_max", "Spacegrp_min", "Sfrac_mean",
                "Sfrac_std", "Sfrac_max", "Sfrac_min", "Pfrac_mean",
                "Pfrac_std", "Pfrac_max", "Pfrac_min", "Dfrac_mean",
                "Dfrac_std", "Dfrac_max", "Dfrac_min", "Ffrac_mean",
                "Ffrac_std", "Ffrac_max", "Ffrac_min", "VEC_mean", "VEC_std",
                "VEC_max", "VEC_min"
            ],
            columns=[name])

        return created_df.T

    def run(self):
        """
        Driver function to run all operations
        and create the feature sheet for ML applications
        """
        rows = self.input_df.iloc[:, 0]

        list_of_elems = []
        for elem in rows:
            list_of_elems.append(elem)

        # create list of poscar files
        self.list_of_poscars = os.listdir(self.poscar_directory_path)

        for poscar_path in self.list_of_poscars:
            # read the poscar file
            read_poscar = read(self.poscar_directory_path + '/' + poscar_path)

            structure = Poscar.from_file(self.poscar_directory_path + '/' +
                                         poscar_path).structure
            nn_algorithm = VoronoiNN(tol=0.25, cutoff=13.0)

            crystal_graph = CrystalGraph(structure, nn_algorithm)
            graph = crystal_graph.graph
            positions = read_poscar.get_positions()

            # get composition of each element in from poscar file
            hea_comp = self.__get_composition(self.poscar_directory_path +
                                              '/' + poscar_path)

            # get distances dictionary
            self.__get_distance(positions, graph)

            list_elem_comp = []
            temp_comp = hea_comp
            elem_comp_rows = temp_comp.iloc[:, 0]
            for comp in elem_comp_rows:
                list_elem_comp.append(comp)

            for _, prop in enumerate(self.prop_list):
                self.__get_elemental_property(read_poscar, graph, positions,
                                              prop, hea_comp, elem_comp_rows,
                                              list_of_elems, rows)

            final_df = pd.DataFrame(
                self.__create_feature_sheet(
                    poscar_path.split("/")[len(poscar_path.split("/")) - 2]))
            self.final_df = pd.concat([self.final_df, final_df])

        # export the created csv
        try:
            self.final_df.to_csv(self.export_path + '/cg_features.csv')
        except OSError as err:
            raise OSError(err) from err

        return ('Find your feature sheet at ' + self.export_path +
                '/cg_features.csv')
