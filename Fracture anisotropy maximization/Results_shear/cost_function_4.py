import numpy as np
from hybrida import *
import collections.abc
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import cm, rc
from hybrida.fem.mesh.elements.element import Element as Element
from hybrida.optimization import ERRMinimization_Multipleloads_2
from matplotlib.colors import LinearSegmentedColormap

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class TOP_cost_function():
    '''
    Class that allows the repeated calculation of the same cost function without having to reinitialize it
    or recompile certain aspects
    '''
    def __init__(self, fracture_weight = 0.5, debug = False, elnums = 25, save_para_view = False, para_view_char = 0):
        if isinstance(fracture_weight, collections.Sequence):
            raise TypeError("Loss weight loss is supposed to be a scalar")
        self.fracture_weight = np.clip(float(fracture_weight), 0, 1)
        
        if isinstance(debug, collections.Sequence):
            raise TypeError("Debug is supposed to be a scalar")
        self.debug = bool(debug)
        
        if isinstance(save_para_view, collections.Sequence):
            raise TypeError("Debug is supposed to be a scalar")
        self.save_para_view = bool(save_para_view)
        
        if isinstance(elnums, collections.Sequence):
            raise TypeError("Element umber loss is supposed to be a scalar")
        self.elnums = max(1, int(elnums)) 
        
        if isinstance(para_view_char, collections.Sequence):
            raise TypeError("Element umber loss is supposed to be a scalar")
        self.para_view_char = max(0, int(para_view_char)) 
        
        
        if not self.debug:
            blockPrint()
        # Define matrix and void materials, assuming linear elastic behaviour
        self.mat_matrix = Material(Elastic(extra=DisplacementType.plane_strain), Young=1, Poisson=0.3)
        
        self.mat_void = Material(Elastic(extra=DisplacementType.plane_strain), Young=1e-6, Poisson=0.3)

        
        # Number of elements in x and y directions
        
        
        # volume constraints
        self.vol1 = 0.501
        self.vol2 = 0.499
        
        # The bound locations (lowerbound is lower left corner)
        self.lowerbound = (0, 0)
        self.upperbound = (1, 1)
        
        # Create original square mesh, without any dynamic remeshing
        self.origMesh_test = Mesh.create('square', 't3', 
                                         (self.elnums, self.elnums), 
                                         self.lowerbound, 
                                         self.upperbound)


        strain1 = np.array([[0, 0.1], [0.0, 0]])
        periodic1 = Periodic("bottom-left", ["bottom-right", "top-left", "top-right"], [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                            ["left", "bottom"], ["right", "top"], [np.array([1, 0]), np.array([0, 1])], strain=strain1)
        self.bc1 = [periodic1]

        # # Uniform strain in the y direction
        strain2 = np.array([[0, 0], [0.1, 0]])
        periodic2 = Periodic("bottom-left", ["bottom-right", "top-left", "top-right"], [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                            ["left", "bottom"], ["right", "top"], [np.array([1, 0]), np.array([0, 1])], strain=strain2)
        self.bc2 = [periodic2]
        



        # Determine fixed nodes, that are not changable.
        self.fixedNodes = []
        
        dist_test = 0.1 * self.origMesh_test.size()
        
        # Get loading original nodes which are set non-design domain during iteration
        for nodeid, coordinates in enumerate(self.origMesh_test.nodes[ndata.coordinates]):

            # For non-design domain in the loading position: Here the force attacks, and material must remain
            # The levelset function here is negativ (which might have to be enforced for random starts)
            # These are in insideKnots as well
            if (coordinates[0] < self.lowerbound[0] + dist_test or 
                coordinates[0] > self.upperbound[0] - dist_test or 
                coordinates[1] < self.lowerbound[1] + dist_test or 
                coordinates[1] > self.upperbound[1] - dist_test):
                self.fixedNodes += [nodeid]

        self.RBFMesh = Mesh.create('rectangular', 'q4', (self.elnums, self.elnums), self.lowerbound, self.upperbound)
    
        rad = 2.5
        self.Ltest = TopOptLevelSetDiscontinuity(None, self.origMesh_test, 
                                                 designVariables = np.zeros((self.elnums + 1) ** 2), 
                                                 name = "interface", 
                                                 fixedNodes = self.fixedNodes, 
                                                 dsp = rad * self.RBFMesh.size(), 
                                                 RBFmesh= self.RBFMesh)
        
        self.bound = self.Ltest.RBF.updatePhi(np.ones_like(self.Ltest.designVariables))
        
        if not self.debug:
            enablePrint()
        # Set LO_prepared to false, so LO can not start until prepare_LO has been run
        self.LO_prepared = False
        # This cost function can remember objective values during local optimization
        self.internal_documentation = True
        
    def get_input_shape(self):
        return (self.elnums + 1, self.elnums + 1)
    
    def get_boundaries(self):
        B = self.bound.reshape(self.elnums + 1, self.elnums + 1)
        return -1.0 * B, 1.0 * B
    
    def get_internal_documentation(self):
        return self.internal_documentation
    
    def get_max_parallel(self):
        return 1
    
    def get_constrain_monitoring(self):
        return 3
        
    def evaluate(self, Input, check_input_size = True):
        '''
        Allows the evaluation of a cost function for a number of samples

        Parameters
        ----------
        Input : numpy.array
            An array with n input samples, which are to be evaluated.
        check_input_size : bool, optional
            A checkmark to determine if input should be check for dimensionality.

        Returns
        -------
        numpy.array
            The n cost function values associated with the inputs.
        '''
        # Check if input is indeed an numpy array
        if not isinstance(Input,np.ndarray):
            raise TypeError("Input is supposed to be a numpy array")
        # Check if input dimensionality matches teh design space
        if Input.size != len(Input) * (self.elnums + 1)**2 and check_input_size:
            raise TypeError("Input dimensions do not fit cost function")
            
        if not self.debug:
            blockPrint()
        
        self.origMesh = Mesh.create('square', 't3', 
                                    (self.elnums, self.elnums), 
                                    self.lowerbound, 
                                    self.upperbound)

        # Get initial values as a function
        Input = Input[0].reshape(-1,1)
        # Apply boundary conditions
        Input[self.fixedNodes, 0] = - 0.1 * self.bound[self.fixedNodes]
        
        rad = 2.5
        self.L1 = TopOptLevelSetDiscontinuity(None, self.origMesh, 
                                              designVariables = Input,
                                              name = "interface", 
                                              fixedNodes = self.fixedNodes, 
                                              dsp = rad * self.RBFMesh.size(), 
                                              RBFmesh= self.RBFMesh,   
                                              firstreinitialize = self.finit)

        
        self.geo = TurboGEngine(self.origMesh, optimization = True)
        self.geo([self.L1], pgroups=[Single("interface", dimension=2), Single("interface", dimension=1)])

        # Analysis
        self.step1 = Static(mesh = self.origMesh,
                      equation = Structural(),
                      bcs = self.bc1,
                      optimization = True,
                      materials = ("surface", self.mat_void, "interface_2D", self.mat_matrix),
                      dofManager = EnrichedManager(self.geo))
        
        self.step2 = Static(mesh = self.origMesh,
                      equation = Structural(),
                      bcs = self.bc2,
                      optimization = True,
                      materials = ("surface", self.mat_void, "interface_2D", self.mat_matrix),
                      dofManager = EnrichedManager(self.geo))

        if self.vectorize:
            self.step1.use_vectorized_t3h = True
            self.step2.use_vectorized_t3h = True
        
        pgroups = ["interface_1D", "top", "right", "left", "bottom"]
        p = 1
        alpha = [0, 0, 0, 0, 0]
        epsilon = [0.01, 0.01, 0.01, 0.01, 0.01]
        
        self.opt = Optimization_LS(steps=[self.step1, self.step2],
                                   objective = Objective(ERRMinimization_Multipleloads(pgroups, p, alpha, epsilon, 
                                                                                       weight = self.fracture_weight)),
                                   constraints = [Constraint(VolumeConstraint_LS(self.vol1)), 
                                                  Constraint(VolumeConstraint_LS(self.vol2, reverse=-1))],
                                   design = self.L1,
                                   optimizer=StijnsMMA(self.L1.designVariables, 
                                                       -1 * np.ones_like(self.L1.designVariables),
                                                       1 * np.ones_like(self.L1.designVariables), 
                                                       move = self.ml, iteration = 0),
                                   convergence = [MaxIteration(0)])
        
        
        if self.save_para_view:
            path = os.getcwd()
            self.opt.out = [Output("mesh", "displacement, element stress", os.path.join(path, 'paraview/test_chocolate_compression_periodic_{}.vtu'.format(self.para_view_char)))]
        self.opt.solve()
        
        
    
        # Return design, history of cost functions, as well as 
        
    
        Frac_rel = np.array(self.opt.objective.function.Gtd1) / np.array(self.opt.objective.function.Gtd0)
        
        G_out = np.concatenate((np.array(self.opt.g[0])[:,np.newaxis,np.newaxis], # upper mass constraint
                                np.array(self.opt.g[1])[:,np.newaxis,np.newaxis], # lower mass constraint
                                Frac_rel[:,np.newaxis,np.newaxis]),               # Toughness coefficient
                               axis = -1)
        if not self.debug:
            enablePrint()
            
        return  (np.array(self.opt.objective.function.objective)[:,np.newaxis], G_out)
        

    
    
         
    def LO(self, Input, num_steps = 10):
        '''
        Cost function which optimizes a number of inputs using Adam

        Parameters
        ----------
        Input : numpy.array
            An array with n input samples, which are to be optimized.
        num_steps : int, optional
            The number of local optimization steps to be done. The default is 10.

        Returns
        -------
        numpy.array
            The optimized input.
        '''
        # Check if Local optimization has been adequately prepared
        if not self.LO_prepared:
            raise AttributeError("Local optimization has not yet been prepared")
        # Check if input is indeed an numpy array
        if not isinstance(Input,np.ndarray):
            raise TypeError("Input is supposed to be a numpy array")
        # Check if input dimensionality matches teh design space
        if Input.size != len(Input) * (self.elnums + 1)**2:
            raise TypeError("Input dimensions do not fit cost function")
        # Check if the number of inputs indeed mathches the number of inputs LO has been prepared for
        if Input.shape[0]!=self.gradient_size:
            raise ValueError("Number of Inputs does not match")
        # Check if number of optimization steps is indeed a scalar
        if isinstance(num_steps, collections.Sequence):
            raise TypeError("Number of LO steps is supposed to be a scalar")
        
        if not self.debug:
            blockPrint()
        
        self.origMesh = Mesh.create('square', 't3', 
                                    (self.elnums, self.elnums), 
                                    self.lowerbound, 
                                    self.upperbound)

        # Get initial values as a function
        # Get initial values as a function
        Input = Input[0].reshape(-1,1)
        # Apply boundary conditions
        Input[self.fixedNodes, 0] = - 0.1 * self.bound[self.fixedNodes]
        
        rad = 2.5
        self.L1 = TopOptLevelSetDiscontinuity(None, self.origMesh, 
                                              designVariables = Input,
                                              name = "interface", 
                                              fixedNodes = self.fixedNodes, 
                                              dsp = rad * self.RBFMesh.size(), 
                                              RBFmesh= self.RBFMesh,   
                                              firstreinitialize = self.finit)

        
        self.geo = TurboGEngine(self.origMesh, optimization = True)
        self.geo([self.L1], pgroups=[Single("interface", dimension=2), Single("interface", dimension=1)])

        # Analysis        
        self.step1 = Static(mesh = self.origMesh,
                      equation = Structural(),
                      bcs = self.bc1,
                      optimization = True,
                      materials = ("surface", self.mat_void, "interface_2D", self.mat_matrix),
                      dofManager = EnrichedManager(self.geo))
        
        self.step2 = Static(mesh = self.origMesh,
                      equation = Structural(),
                      bcs = self.bc2,
                      optimization = True,
                      materials = ("surface", self.mat_void, "interface_2D", self.mat_matrix),
                      dofManager = EnrichedManager(self.geo))
        
        if self.vectorize:
            self.step1.use_vectorized_t3h = True
            self.step2.use_vectorized_t3h = True
            
            
        pgroups = ["interface_1D", "top", "right", "left", "bottom"]
        p = 1
        alpha = [0, 0, 0, 0, 0]
        epsilon = [0.01, 0.01, 0.01, 0.01, 0.01]
        
        self.opt = Optimization_LS(steps=[self.step1, self.step2],
                                   objective = Objective(ERRMinimization_Multipleloads(pgroups, p, alpha, epsilon, 
                                                                                       weight = self.fracture_weight)),
                                   constraints = [Constraint(VolumeConstraint_LS(self.vol1)), 
                                                  Constraint(VolumeConstraint_LS(self.vol2, reverse=-1))],
                                   design = self.L1,
                                   optimizer=StijnsMMA(self.L1.designVariables, 
                                                       -1 * np.ones_like(self.L1.designVariables),
                                                       1 * np.ones_like(self.L1.designVariables), 
                                                       move = self.ml, iteration = num_steps),
                                   convergence = [MaxIteration(num_steps)])
        
        if self.save_para_view:
            path = os.getcwd()
            self.opt.out = [Output("mesh", "displacement, element stress", os.path.join(path, 'paraview/test_chocolate_compression_periodic_{}.vtu'.format(self.para_view_char)))]
        self.opt.solve()
        
        Frac_rel = np.array(self.opt.objective.function.Gtd1) / np.array(self.opt.objective.function.Gtd0)
        
        G_out = np.concatenate((np.array(self.opt.g[0])[:,np.newaxis,np.newaxis], # upper mass constraint
                                np.array(self.opt.g[1])[:,np.newaxis,np.newaxis], # lower mass constraint
                                Frac_rel[:,np.newaxis,np.newaxis]),               # Toughness coefficient
                               axis = -1)
        if not self.debug:
            enablePrint()
        # Return design, history of cost functions, as well as 
        return  (self.L1.levelSetValues.reshape(1, self.elnums + 1, self.elnums + 1), 
                 np.array(self.opt.objective.function.objective)[:,np.newaxis],
                 G_out)
    
    def prepare_LO(self, num_samples = 1, ml = 0.001, firstreinitialize = True, vectorize = True):
        '''
        Add the possibility for local optimization on this specific costfuntion, using Adam

        Parameters
        ----------
        num_samples : int, optional
            Number of samples which should be optimized in parallel. The default is 1.
        alpha : float, optional
            Adam learning rate. The default is 0.001.
        beta_m : float, optional
            Adam momentum factor 1. The default is 0.9.
        beta_v : float, optional
            Adam momentum factor 2. The default is 0.999.
        verbose : int, optional
            Determines if intermediate steps are written to console. 0 means no output,
            while 1 stands for output with progress bar and 2 for output without a
            progress bar. The default is 0.

        '''
        # check if number smaples is actually a scalar
        if isinstance(num_samples, collections.Sequence):
            raise TypeError("Num samples is supposed to be a scalar")
        self.gradient_size = num_samples
        
        if isinstance(ml, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.ml = np.clip(float(ml), 0, None)
        
        if isinstance(firstreinitialize, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.finit = bool(firstreinitialize)
        
        if isinstance(vectorize, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.vectorize = bool(vectorize)
        
        # During multiple call to self.LO during a singel optimization run, it is not necessary to overwrite 
        # the weights in the SIMPLE layer, as they stay the same
        # only after initialization or  reseting is overwriting really needed
        self.LO_prepared = True
        
        
    def LO_reset_state(self, ml = 0.001, firstreinitialize = True, vectorize = True):
        if not self.LO_prepared:
            raise AttributeError("Local optimization has not yet been prepared")
        
        if isinstance(ml, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.ml = np.clip(float(ml), 0, None)

        if isinstance(firstreinitialize, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.finit = bool(firstreinitialize)
        
        if isinstance(vectorize, collections.Sequence):
            raise TypeError("Volumen constraint is supposed to be a scalar")
        self.vectorize = bool(vectorize)
        
        
    def create_picture(self, Input, savefile = None, figsize = (5,5)):
        # Check if input is indeed an numpy array
        if not isinstance(Input,np.ndarray):
            raise TypeError("Input is supposed to be a numpy array")
        # Check if input dimensionality matches teh design space
        if np.prod(Input.shape[-2:]) != (self.elnums + 1)**2:
            raise TypeError("Input dimensions do not fit cost function")
        # Check if the number of inputs indeed mathches the number of inputs LO has been prepared for

        origMesh = Mesh.create('square', 't3', (self.elnums, self.elnums), 
                               self.lowerbound, 
                               self.upperbound)

        # Get initial values as a function
        Input = Input.reshape(-1, 1)
        
        rad = 2.5
        L1 = TopOptLevelSetDiscontinuity(None, origMesh, 
                                         designVariables = Input, 
                                         name = "interface", 
                                         fixedNodes = self.fixedNodes, 
                                         dsp = rad * self.RBFMesh.size(), 
                                         RBFmesh= self.RBFMesh, 
                                         firstreinitialize = False)
        
        
        geo = TurboGEngine(origMesh, optimization = True)
        geo([L1], pgroups=[Single("interface", dimension=2), Single("interface", dimension=1)])
        
        nodes = origMesh.nodes[ndata.coordinates]
                
        conn1 = np.array(origMesh.elements[origMesh.d]['t3'][edata.connectivity])
        temp = Input[conn1, 0] > 0
        conn1 = conn1[temp.all(1) | np.invert(temp).all(1), :]
        conn2 = np.array(origMesh.elements[origMesh.d]['t3h'][edata.connectivity])
        
        
        # plt.gcf().set_facecolor("white")

        patches = []
        colors = []
        linecolors = []
        lines = []
        bordercolors = []
        border = []

        for etype, id, conn, egroup in origMesh.elements(origMesh.d, edata=[edata.connectivity, edata.pgroup]):
            if conn is not None:
                if egroup == 9:
                    colors.append(1)
                else:
                    colors.append(0)
                polygon = Polygon(origMesh.nodes[ndata.coordinates][conn])
                patches.append(polygon)
                    
        

        for etype, id, conn in origMesh.elements(origMesh.d-1, edata=[edata.connectivity]):
            if conn is not None and etype == "l2h":
                bordercolors.append(1)
                border.append(origMesh.nodes[ndata.coordinates][conn])
            
            if conn is not None and etype == "l2":
                linecolors.append(1)
                lines.append(origMesh.nodes[ndata.coordinates][conn])
                


        bcdict = {'red': ((0, 0.0, 0.0),
                          (1, 0.0, 0.0)),
               'green': ((0, 0.0, 0.0),
                         (1, 0.0, 0.0)),
               'blue': ((0, 0.0, 0.0),
                        (1, 0.0, 0.0))}
    
        bcmap = LinearSegmentedColormap('custom_cmap_l', bcdict)
        
        lcdict = {'red': ((0, 0.75, 0.75),
                          (1, 0.75, 0.75)),
               'green': ((0, 0.75, 0.75),
                         (1, 0.75, 0.75)),
               'blue': ((0, 0.75, 0.75),
                        (1, 0.75, 0.75))}
    
        lcmap = LinearSegmentedColormap('custom_cmap_l', lcdict)
        
        fig = plt.figure(figsize = figsize, frameon=False)
        ax = fig.add_subplot(111)
        ax.autoscale(enable=True, axis='both')
        ax.set_aspect('equal', adjustable='box') 
        plt.gcf().set_facecolor("white")

        
        p = PatchCollection(patches, cmap=cm.gray_r, alpha=0.5, linewidths=(2,))
        p.set_array(np.asarray(colors))
        ax.add_collection(p)
        
        l = LineCollection(lines, cmap=lcmap, linewidths=0.5)
        l.set_array(np.asarray(linecolors))
        ax.add_collection(l)
        
        b = LineCollection(border, cmap=bcmap, linewidths=1)
        b.set_array(np.asarray(bordercolors))
        ax.add_collection(b)
        
        fig.tight_layout()
        plt.axis('off')
        plt.margins(0.01, 0.01)
        plt.tight_layout()
        plt.show() 
        if savefile is not None:
            file = savefile + '.pdf'
            fig.savefig(file, bbox_inches='tight', pad_inches=0)
        
        
        
        
        
        