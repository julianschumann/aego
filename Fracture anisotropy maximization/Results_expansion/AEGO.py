import numpy as np
import collections.abc
import tensorflow as tf
import tensorflow.keras.backend as K
import time

class AEGO():
    '''
    Class that implements the AEGO method
    '''
    def __init__(self, cost_function):
        '''
        Initialize the bare optimization algorithm

        Parameters
        ----------
        cost_function :
            The cost function to be optimized using the AEGO algorithm
        '''
        # Check if the loaded cost function has the required attributes
        if not all(hasattr(cost_function, attr) for attr in ["get_input_shape", "get_boundaries", 
                                                             "get_internal_documentation", "get_max_parallel",
                                                             "get_constrain_monitoring", "evaluate", 
                                                             "prepare_LO", "LO", "LO_reset_state"]):
            raise TypeError("This cost function class lacks the required methods")

        self.cost = cost_function
        
        self.max_num_samples = self.cost.get_max_parallel()
        
        self.dim = self.cost.get_input_shape()
        
        
        # Get boundaries of search space (hyperrectangle),
        # which can be scalars or arrays.
        self.x_min_value, self.x_max_value = self.cost.get_boundaries()
        
        if isinstance(self.x_min_value, collections.Sequence):
            self.x_min_value = np.array(self.x_min_value)
            if self.x_min_value.shape != self.dim:
                raise TypeError("Wrong dimension for lower search space limits.")
        else:
            self.x_min_value = self.x_min_value * np.ones(self.dim)
            
        
        if isinstance(self.x_max_value, collections.Sequence):
            self.x_max_value = np.array(self.x_max_value)
            if self.x_max_value.shape != self.dim:
                raise TypeError("Wrong dimension for lower search space limits.")
        else:
            self.x_max_value = self.x_max_value * np.ones(self.dim)
            
        self.x_interval = self.x_max_value - self.x_min_value
        
        # Check if for local optimization each step has to be initiated by the 
        # AEGO algorithm, or if the cost function can do that internally and then return all
        # cost function values during optimization as weel
        self.documentation = 'AEGO'
        if self.cost.get_internal_documentation():
            self.documentation = 'cost'
            
        self.num_constraints = int(self.cost.get_constrain_monitoring())
                
    def generate_samples(self, num_samples = 1000, LO_params = None,
                         lam = 100, deflation = False, num_deflation_starts = 30, factor_deflate = 0.0,
                         use_MPI = False, mpi_comm = None, mpi_rank = 0, mpi_size = 1):
        '''        
        Allows for generation of the training samples for the autoencoder.
        This is the first step of the AEGO-algorithm

        Parameters
        ----------
        num_samples : int, optional
            Number of training samples to create. The default is 1000.
        verbose : int, optional
            
        LO_params : dictionary, optional
            Parameters that are used by the local optimizer, like for example Adam. Defualt is None.
        lam : int, optional
            Number of local optimization iterations. The default is 100.
        deflation : bool, optional
            Determine if deflation should be used in an attempt to increase the variance
            in the training samples. The default is False.
        num_deflation_starts : int, optional
            If deflation is used, this is the number of independent starting points. The default is 30.
        use_MPI : bool, optional
            Check if a mulitple processors should be used. The default is False
        mpi_comm : mpi4py.MPI.Intracomm, optional.
            Object allowing for communication between separate processes. The default is None.
        mpi_rank : int, optional
            ID of process used. The default is 0.
        mpi_size : int, optional
            Number of processes used. The default is 1.

        Returns
        -------
        X_rand : numpy.ndarray
            Locally optimized training samples.
        C_rand : numpy.ndarray
            History of cost function for generation of training samples.

        '''
            
        # Check if num_samples is a scalar
        if isinstance(num_samples, collections.Sequence):
            raise TypeError("Number samples is supposed to be a scalar")
        num_samples = max(0,int(num_samples))  
        # Check if lam is an scalar
        if isinstance(lam, collections.Sequence):
            raise TypeError("Number of LO steps is supposed to be a scalar")
        # Ensure that lam is positive
        lam = max(int(lam),0)
        deflation = bool(deflation)

        # check if delfation is to be used
        if deflation == False:
            # Determine the number of batches needed to run all samples
            num_batches_perrank = int(np.ceil(num_samples / (self.max_num_samples * mpi_size)))
            # Calculate the interger number of samples in each batch
            num_batch_samples = int(np.floor( num_samples / (num_batches_perrank * mpi_size)))
            # Calculate the number of samples that results from an integer number of batches
            # with an interger number of samples each
            num_samples_perrank = num_batches_perrank * num_batch_samples
            if mpi_rank == 0:
                print('Actual number of samples generated: {}/{}'.format(num_samples_perrank * mpi_size, num_samples))
            
            # Determine the samples belonging to each batch
            batch_index = np.arange(num_samples_perrank).reshape(num_batches_perrank, num_batch_samples)
            
            # Set up the cost function with the spicific number of batch samples
            self.cost.prepare_LO(num_samples = num_batch_samples, **LO_params)
            
            
            if mpi_rank == 0:
                print('Generating samples - iteration {}'.format(0))
            # Generate initial samples
            X_rand_rank = np.random.rand(*([num_samples_perrank] + list(self.dim))) * self.x_interval + self.x_min_value
            X_rand_rank_init = X_rand_rank.copy()
            # calculate the initial cost function values
            C_rand_rank = np.zeros((lam + 1, num_samples_perrank))
            if self.num_constraints > 0:
                G_rand_rank = np.zeros((lam + 1, num_samples_perrank, self.num_constraints)) 
            
            if self.documentation == 'AEGO':
                for i in range(num_batches_perrank):
                    if self.num_constraints > 0:
                        (C_rand_rank[0,batch_index[i]], 
                         G_rand_rank[0,batch_index[i],:]) = self.cost.evaluate(X_rand_rank[batch_index[i],:])  
                    else:
                        C_rand_rank[0,batch_index[i]] = self.cost.evaluate(X_rand_rank[batch_index[i],:])
                    
                    self.cost.LO_reset_state(**LO_params)
                    for g in range(1, lam + 1):
                        if mpi_rank == 0:
                            print('Generating samples - iteration {} - batch {}'.format(g,i))
                        X_rand_rank[batch_index[i],:] = self.cost.LO(X_rand_rank[batch_index[i],:], 1)
                        if self.num_constraints > 0:
                            (C_rand_rank[g,batch_index[i]], 
                             G_rand_rank[g,batch_index[i],:]) = self.cost.evaluate(X_rand_rank[batch_index[i],:])  
                        else:
                            C_rand_rank[g,batch_index[i]] = self.cost.evaluate(X_rand_rank[batch_index[i],:])   
            else:
                # iterate for each step
                for i in range(num_batches_perrank):
                    # Reset optimizer params if necessary (for example momentum)
                    self.cost.LO_reset_state(**LO_params)
                    if mpi_rank == 0:
                        print('Generating samples - {} iterations - batch {}'.format(lam,i))                    
                    if self.num_constraints > 0:
                        (X_rand_rank[batch_index[i],:],
                         C_rand_rank[:, batch_index[i]],
                         G_rand_rank[:, batch_index[i],:]) = self.cost.LO(X_rand_rank[batch_index[i],:], lam)
                    else:
                        (X_rand_rank[batch_index[i],:],
                         C_rand_rank[:, batch_index[i]]) = self.cost.LO(X_rand_rank[batch_index[i],:], lam)
        
        else:
            # Check if given cost function allows for deflation adaption
            if not all(hasattr(self.cost, attr) for attr in ["LO_deflation", "prepare_LO_deflation", "LO_deflation_reset_state"]):
                raise AttributeError("Deflation is not possible for this cost function implementation")
            # Check if number of delfation starts is a scalar
            if isinstance(num_deflation_starts, collections.Sequence):
                raise TypeError("Number deflation starts is supposed to be a scalar")
            # Check if deflation weight is a scalar
            if isinstance(factor_deflate, collections.Sequence):
                raise TypeError("Number deflation starts is supposed to be a scalar")
            # Ensure num_deflation_starts is positive integer
            num_deflation_starts = max(0,int(num_deflation_starts)) 
            # Calculate number of batches for deflation starts
            num_deflation_batches_perrank = int(np.ceil(num_deflation_starts / (self.max_num_samples * mpi_size)))
            # Calculate the interger number of samples in each batch
            num_deflation_batch_samples = int(np.floor( num_deflation_starts / (num_deflation_batches_perrank * mpi_size)))
            # Calculate the number of samples that results from an integer number of batches
            # with an interger number of samples each
            num_deflation_starts_perrank = num_deflation_batches_perrank * num_deflation_batch_samples
            if mpi_rank == 0:
                print('Actual number of deflation start points used: {}/{}'
                      .format(num_deflation_starts_perrank * mpi_size, num_deflation_starts))
            
            # Determine the samples belonging to each batch
            deflation_batch_index = np.arange(num_deflation_starts_perrank).reshape(num_deflation_batches_perrank, 
                                                                                    num_deflation_batch_samples)
            
            # Set up the cost function with the spicific number of batch samples
            self.cost.prepare_LO_deflation(factor_deflate = factor_deflate, num_samples = num_deflation_batch_samples, **LO_params)
            
            # Determine number of delfation steps needed
            num_deflation_steps = int(np.floor(num_samples / (num_deflation_starts_perrank * mpi_size)))
            # Determine actual number of samples created
            num_samples_perrank = num_deflation_starts_perrank * num_deflation_steps
            if mpi_rank == 0:
                print('Actual number of samples used: {}/{}'.format(num_samples_perrank * mpi_size, num_samples))
            
            
            # intialize samples
            X_rand_rank = np.random.rand(*([num_deflation_starts_perrank] + list(self.dim))) * self.x_interval + self.x_min_value
            X_rand_rank_init = X_rand_rank.copy()
            # Calculate initial results
            C_rand_rank = np.zeros((lam + 1, num_deflation_starts_perrank))
            if self.num_constraints > 0:
                G_rand_rank = np.zeros((lam + 1, num_deflation_starts_perrank, self.num_constraints))
            
            
            
            if self.documentation == 'AEGO':         # Optimize initial samples:
                for i in range(num_deflation_batches_perrank):
                    if self.num_constraints > 0:
                        (C_rand_rank[0,deflation_batch_index[i]], 
                         G_rand_rank[0,deflation_batch_index[i],:]) = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],:])  
                    else:
                        C_rand_rank[0,deflation_batch_index[i]] = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],:])
                # Use initial position for every step
                X_rand_rank = np.repeat(X_rand_rank[:,np.newaxis,:], num_deflation_steps, axis=1)
                X_rand_rank_init = np.repeat(X_rand_rank_init[:,np.newaxis,:], num_deflation_steps, axis=1)
                C_rand_rank = np.repeat(C_rand_rank[:,:,np.newaxis], num_deflation_steps, axis=2)
                if self.num_constraints > 0:
                    G_rand_rank = np.repeat(G_rand_rank[:,:,np.newaxis,:], num_deflation_steps, axis=2)
                                            
                                            
                if mpi_rank == 0:
                    print('Generating samples - deflation step {}'.format(0))
                 
                for i in range(num_deflation_batches_perrank): 
                    # Reset optimizer params if necessary (for example momentum)  
                    self.cost.LO_deflation_reset_state(**LO_params)
                    for g in range(1, lam + 1):
                        if mpi_rank == 0:
                            print('Generating samples - deflation step {} - iteration {} - batch {}'.format(0,g,i))
                        X_rand_rank[deflation_batch_index[i],0,:] = self.cost.LO(X_rand_rank[deflation_batch_index[i],0,:], 1)
                        
                        if self.num_constraints > 0:
                            (C_rand_rank[g,deflation_batch_index[i],0], 
                             G_rand_rank[g,deflation_batch_index[i],0,:]) = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],0,:]) 
                        else:
                            C_rand_rank[g,deflation_batch_index[i],0] = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],0,:])
                    
                    
                # Determine for how many local optimization steps the deflation should be used
                # so local convergence is still possible afterwards
                intermediate = int(0.75 * lam)
                
                for b in range(1, num_deflation_steps):
                    if mpi_rank == 0:   
                        print('Generating samples - deflation step {}'.format(b))
                    for i in range(num_deflation_batches_perrank):    
                        # Reset optimizer params if necessary (for example momentum)
                        self.cost.LO_deflation_reset_state(**LO_params)
                        
                        for g in range(1, intermediate):
                            if mpi_rank == 0:
                                print('Generating samples - deflation step {} - iteration {} - batch {}'.format(b,g,i))
                            X_rand_rank[deflation_batch_index[i],b,:] = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                               X_rand_rank[deflation_batch_index[i],:b,:], 1)
                            if self.num_constraints > 0:
                                (C_rand_rank[g,deflation_batch_index[i],b], 
                                 G_rand_rank[g,deflation_batch_index[i],b,:]) = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],b,:]) 
                            else:
                                C_rand_rank[g,deflation_batch_index[i],b] = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],b,:]) 
                      
                        for g in range(intermediate, lam + 1): 
                            if mpi_rank == 0:
                                print('Generating samples - deflation step {} - iteration {} - batch {}'.format(b,g,i))
                            X_rand_rank[deflation_batch_index[i],b,:] = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                               X_rand_rank[deflation_batch_index[i],:0,:], 1)
                            if self.num_constraints > 0:
                                (C_rand_rank[g,deflation_batch_index[i],b], 
                                 G_rand_rank[g,deflation_batch_index[i],b,:]) = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],b,:]) 
                            else:
                                C_rand_rank[g,deflation_batch_index[i],b] = self.cost.evaluate(X_rand_rank[deflation_batch_index[i],b,:]) 
                                
            else:
                # Use initial position for every step
                X_rand_rank = np.repeat(X_rand_rank[:,np.newaxis,:], num_deflation_steps, axis=1)
                X_rand_rank_init = np.repeat(X_rand_rank_init[:,np.newaxis,:], num_deflation_steps, axis=1)
                C_rand_rank = np.repeat(C_rand_rank[:,:,np.newaxis], num_deflation_steps, axis=2)
                if self.num_constraints > 0:
                    G_rand_rank = np.repeat(G_rand_rank[:,:,np.newaxis,:], num_deflation_steps, axis=2)
                
                if mpi_rank == 0:
                    print('Generating samples - deflation step {}'.format(0))
                 
                for i in range(num_deflation_batches_perrank): 
                    # Reset optimizer params if necessary (for example momentum)  
                    self.cost.LO_deflation_reset_state(**LO_params)
                    
                    if mpi_rank == 0:
                        print('Generating samples - deflation step {} - {} iterations - batch {}'.format(0,lam,i))
                    if self.num_constraints > 0:
                        (X_rand_rank[deflation_batch_index[i],0,:],
                         C_rand_rank[:,deflation_batch_index[i],0],
                         G_rand_rank[:,deflation_batch_index[i],0,:]) = self.cost.LO(X_rand_rank[deflation_batch_index[i],0,:], lam)
                    else:
                        (X_rand_rank[deflation_batch_index[i],0,:],
                         C_rand_rank[:,deflation_batch_index[i],0]) = self.cost.LO(X_rand_rank[deflation_batch_index[i],0,:], lam)
            
                # Determine for how many local optimization steps the deflation should be used
                # so local convergence is still possible afterwards
                intermediate = int(0.75 * lam)
                for b in range(1, num_deflation_steps):
                    if mpi_rank == 0:   
                        print('Generating samples - deflation step {}'.format(b))
                    for i in range(num_deflation_batches_perrank):    
                        # Reset optimizer params if necessary (for example momentum)
                        self.cost.LO_deflation_reset_state(**LO_params)
                        if mpi_rank == 0:
                            print('Generating samples - {} iterations - batch {}'.format(lam,i))
                                                                                                            
                        if self.num_constraints > 0:
                            (X_rand_rank[deflation_batch_index[i],b,:],
                             C_rand_rank[: intermediate,deflation_batch_index[i],b],
                             G_rand_rank[: intermediate,deflation_batch_index[i],b,:]) = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                                                  X_rand_rank[deflation_batch_index[i],:b,:], intermediate - 1)
                            
                            (X_rand_rank[deflation_batch_index[i],b,:],
                             C_rand_rank[intermediate - 1:,deflation_batch_index[i],b],
                             G_rand_rank[intermediate - 1:,deflation_batch_index[i],b,:]) = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                                               X_rand_rank[deflation_batch_index[i],:0,:], lam + 1 - intermediate)
                        else:
                            (X_rand_rank[deflation_batch_index[i],b,:],
                             C_rand_rank[: intermediate,deflation_batch_index[i],b]) = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                                                X_rand_rank[deflation_batch_index[i],:b,:], intermediate - 1)                                                                              
                      
                            (X_rand_rank[deflation_batch_index[i],b,:],
                             C_rand_rank[intermediate - 1:,deflation_batch_index[i],b]) = self.cost.LO_deflation(X_rand_rank[deflation_batch_index[i],b,:], 
                                                                                                                     X_rand_rank[deflation_batch_index[i],:0,:], lam + 1 - intermediate)
                    
                    
            # Unroll C_rand and X_rand
            X_rand_rank = X_rand_rank.reshape([num_samples_perrank] + list(self.dim))
            X_rand_rank_init = X_rand_rank_init.reshape([num_samples_perrank] + list(self.dim))
            C_rand_rank = C_rand_rank.reshape(lam + 1, num_samples_perrank)
            if self.num_constraints > 0:
                G_rand_rank = G_rand_rank.reshape(lam + 1, num_samples_perrank, self.num_constraints)
        
        # output the final training samples as well cost function history over run
        if self.num_constraints > 0:
            return X_rand_rank, X_rand_rank_init, C_rand_rank, G_rand_rank
        else:
            return X_rand_rank, X_rand_rank_init, C_rand_rank
    
    def pretrain_Decoder(self, Encoder_list, Decoder_list, X_train, epochs = 100, batch_size = 20,
                         use_MPI = False, mpi_comm = None, mpi_rank = 0, mpi_size = 1):
        '''
        A function that allows the pretraining of stacked encoder and decoder before 
        the second step of AEGO.

        Parameters
        ----------
        Encoder_list : list
            A list of the encoder layers. This goes from outside to inside.
        Decoder_list : list
            A list of the decoder layers. This goes from outside to inside.
        X_train : numpy.ndarray
            Training samples used for training.
        epochs : int, optional
            Number of epochs for traiing the autoencoder. The default is 100.
        batch_size : int, optional
            Size of batches for autoencoder training. The default is 20.

        Returns
        -------
        keras.Model
            Encoder network of a pretrained utoencoder.
        keras.Model
            Decoder netwrok of a pretrained autoencoder.
        float
            The mean squared reconstruction loss of the autoencoder.

        '''
        if not (isinstance(Decoder_list,list) or isinstance(Decoder_list,np.ndarray)):
            raise TypeError("A list of decoder components was expected")
        if not (isinstance(Encoder_list,list) or isinstance(Encoder_list,np.ndarray)):
            raise TypeError("A list of encoder components was expected")
        if len(Decoder_list) != len(Encoder_list):
            raise TypeError("Equal number of encoder and decoder components are necessary")
        
        DL = Decoder_list
        EL = Encoder_list
        num_parts = len(Decoder_list)
        
        for i in range(num_parts):
            # Lists go from outside to inside
            # check if each part is a neural network model
            if not isinstance(DL[i], tf.keras.models.Model):
                raise TypeError("Decoder parts must be a keras model")
            if not isinstance(EL[i], tf.keras.models.Model):
                raise TypeError("Encoder parts must be a keras model")
            # check if the model parts fit together one after another
            if i>0:
                if EL[i].input_shape != EL[i-1].output_shape:
                    raise TypeError("Encoder parts to not match")
            else:
                if EL[i].input_shape[1:] != self.dim:
                    raise ValueError("Autoencoder search space does not match cost function")
            # Check if corresponding decoder and encoder parts fit
            if (EL[i].input_shape != DL[i].output_shape) or (EL[i].output_shape != DL[i].input_shape):
                    raise TypeError("Encoder and decoder shapes do not match")
        # check if training data in indeed an numpy array
        if not isinstance(X_train, np.ndarray):
            raise TypeError("Training data is supposed to be a numpy array")
        # Check if input dimensionality matches the design space
        if X_train.shape[1:] != self.dim:
            raise TypeError("Training data dimensions do not fit cost function")
        # Ensure epochs can be made into list of positive integers
        if isinstance(epochs, collections.Sequence):
            if len(epochs) != len(DL):
                raise TypeError("Epoch number is supposed to be a scalar or list with lenght equal to that of encoder")
            else:
                epochs = list(epochs)
                for i in range(len(epochs)):
                    if isinstance(epochs[i], collections.Sequence):
                        raise TypeError("Number of epochs needs to be a scalar at position {}".format(i))
                    else:
                        epochs[i] = int(max(0, epochs[i]))
        else:
            epochs = [int(max(0, epochs))] * len(DL)
        # Ensure batch_size is an positive integer
        if isinstance(batch_size, collections.Sequence):
            raise TypeError("Batch size is supposed to be a scalar")
        batch_size = int(max(0, batch_size))
        
        # set X_train to domain [0,1]
        X_train = (X_train - self.x_min_value) / self.x_interval
        
        search_space = tf.keras.layers.Input(self.dim)
        
        for i in range(num_parts):
            if mpi_rank == 0:
                print('Training layer {}/{}'.format(i + 1, num_parts))
            latent_space = tf.keras.layers.Input(EL[i].output_shape[1:])
            
            # Add inner parts to encoder and decoder, and make only them trainable
            if i==0:
                Encoder_help = tf.keras.models.Model(search_space, EL[i](search_space))
                Decoder_help = tf.keras.models.Model(latent_space, DL[i](latent_space))
            else:
                Encoder_help.trainable=False
                Decoder_help.trainable=False
                Encoder_help = tf.keras.models.Model(search_space, EL[i](Encoder_help(search_space)))
                Decoder_help = tf.keras.models.Model(latent_space, Decoder_help(DL[i](latent_space)))
                
            AE_help = tf.keras.models.Model(search_space, Decoder_help(Encoder_help(search_space)))
            
            # Train innner parts of the autoencoder
            AE_help.compile(tf.keras.optimizers.Adam(learning_rate=0.0002), 'mse')
            AE_help.fit(X_train, X_train, epochs = epochs[i], batch_size = batch_size, verbose = 2)
        
        # make whole autoencoder trainable again
        for i in range(num_parts):
            if i==0:
                enc = EL[i](search_space)
                dec = DL[num_parts - i - 1](latent_space)
            else:                
                enc = EL[i](enc)
                dec = DL[num_parts - i - 1](dec)
                
        Encoder = tf.keras.models.Model(search_space, enc)
        Decoder = tf.keras.models.Model(latent_space, dec)
        
        
        Encoder.trainable = True
        Decoder.trainable = True     
        
        # This might cause memory issues, so if the one step prediction fail,
        # one instead calculates the loss batch wise
        try:
            Z_train = Encoder.predict(X_train, batch_size = batch_size)
        except:
            Z_train = np.zeros((len(X_train), *Encoder.output_shape[1:]))
            for i in range(len(X_train)):
                Z_train[[i],:] = Encoder.predict(X_train[[i],:])          
        
        try:
            X_train_pred = Decoder.predict(Z_train, batch_size = batch_size)
        except:
            X_train_pred = np.zeros(X_train.shape)
            for i in range(len(X_train)):
                X_train_pred[[i],:] = Decoder.predict(Z_train[[i],:])

        final_loss = np.mean((X_train - X_train_pred)**2)
        
        return Encoder, Decoder, final_loss
    
    def train_Decoder(self, Encoder, Decoder, X_train, epochs = 100, batch_size = 20, 
                      use_Surr = False, C_train = None, Surr_lam = 0.0, use_AAE = False, ad_prob = 0.0,
                      use_MPI = False, mpi_comm = None, mpi_rank = 0, mpi_size = 1):
        '''
        Allows the generation of an Decoder by training an Autoencoder on some training samples.
        This is the second step of the AEGO-algorithm

        Parameters
        ----------
        Encoder : keras.Model
            The encoder network, which together with the decoder will form an autoencoder.
        Decoder : keras.Model
            The decoder network used to project latent space samples into the search space.
            Encoder and Decoder might have been pretrained.
        X_train : numpy.ndarray
            The training samples on which the autoencoder will be trained.
        epochs : int, optional
            Number of epochs for traiing the autoencoder. The default is 100.
        batch_size : int, optional
            Size of batches for autoencoder training. The default is 20.
        use_Surr : bool, optional
            The decision if a surrogate network is used during training to regularize the autoencoder. The default is False.
        C_train : numpy.ndarry, optional
            The cost function values corresponding to the training samples,
            necessary if surrogate network is used. The default is None.
        Surr_lam : float, optional
            If the surrogate network is used, this is the weight attached to this regularizer. The default is 0.0.
        use_AAE : bool, optional
            The decision if an discriminator should be added to form an adversarial autoencoder. The default is False.
        ad_prob : float, optional
            The probability that discriminator loss function is used in each epoch. The default is 0.0.


        Returns
        -------
        keras.Model
            Encoder network of a pretrained utoencoder.
        keras.Model
            Decoder netwrok of a pretrained autoencoder.
        float
            The mean squared reconstruction loss of the autoencoder.

        '''
        # Check if decoder is a keras model
        if not isinstance(Decoder, tf.keras.models.Model):
            raise TypeError("Decoder must be a keras model")
        if not isinstance(Encoder, tf.keras.models.Model):
            raise TypeError("Encoder must be a keras model")
        # Check if decoder maps into the right design space
        if Decoder.output_shape[1:] != self.dim:
            raise ValueError("Decoder does not project into design_space")
        # Check if encoder and decoder match
        if Decoder.output_shape[1:] != Encoder.input_shape[1:]:
            raise ValueError("Encoder and Decoder do not share same search space")
        if Decoder.input_shape[1:] != Encoder.output_shape[1:]:
            raise ValueError("Encoder and Decoder do not share same latent space")
        if not isinstance(X_train, np.ndarray):
            raise TypeError("Training data is supposed to be a numpy array")
        # Check if training samples dimensionality matches the search space
        if X_train.shape[1:] != self.dim:
            raise TypeError("Training data dimensions do not fit cost function")
        # check if epcohs is a positive integer
        if isinstance(epochs, collections.Sequence):
            raise TypeError("Epoch number is supposed to be a numpy array")
        epochs = int(max(0, epochs))
        # check if batch size is a positive integer
        if isinstance(batch_size, collections.Sequence):
            raise TypeError("Batch size is supposed to be a numpy array")
        batch_size = int(max(0, batch_size))
        
        # Add elements to objects (and set X_train to domain [0,1])
        X_train = (X_train - self.x_min_value) / self.x_interval
        C_train = C_train
        Decoder = Decoder
        self.edim = int(Decoder.input_shape[1])
        Encoder = Encoder
        
        # Designs inputs for search space, latent space, and cost function space
        search_space = tf.keras.layers.Input(self.dim)
        latent_space = tf.keras.layers.Input(self.edim)
        
        # Build simple autoencoder out of encoder and decoder
        AE = tf.keras.models.Model(search_space, Decoder(Encoder(search_space)))
        
        # Check if surrogate model will be used
        if use_Surr:
            # Check if training data cost is an array
            if not isinstance(C_train, np.ndarray):
                raise TypeError("Training data output is supposed to be a numpy array")
            # Check if every training sample has a cost function value assigned
            if len(X_train) != len(C_train):
                raise TypeError("Training data and its output do not match in number")
            # Normalize cmin to be within surrogate network output range
            C_train = 0.8 * C_train / (np.max(C_train) - np.min(C_train)) + 0.1
            
            # Check if surrogate network loss weighting is a positive scalar
            if isinstance(Surr_lam, collections.Sequence):
                raise TypeError("Surrogate weight is supposed to be a scalar")
            Surr_lam = tf.constant(float(max(Surr_lam, 0)))
            
            cost_space = tf.keras.layers.Input(1)
            # Build surrogate network
            surr = tf.keras.layers.Dense(self.edim, activation='sigmoid')(latent_space)
            surr = tf.keras.layers.Dense(self.edim, activation='sigmoid')(surr)
            surr = tf.keras.layers.Dense(self.edim, activation='sigmoid')(surr)
            surr = tf.keras.layers.Dense(self.edim, activation='sigmoid')(surr)
            surr = tf.keras.layers.Dense(1, activation='sigmoid')(surr)
            
            # Integrate loss function into network (squared error)
            surr = tf.subtract(surr,cost_space)
            surr = K.square(surr)
            surr = tf.math.multiply(surr,Surr_lam)
            
            # Extract decoder from latent space
            ae_surr = Decoder(latent_space)
            # Add reconstruction loss to decoder (mean squared error)
            ae_surr = tf.subtract(ae_surr,search_space)
            ae_surr = K.square(ae_surr)
            ae_surr = K.mean(ae_surr, axis=np.arange(1, X_train.ndim), keepdims = False)
            ae_surr = K.expand_dims(ae_surr, axis = 1)
            
            # Combine the two outputs (L_s(z) and L_rec(z)), using the weight 
            ae_surr = tf.keras.layers.add([ae_surr, surr])
            
            # Build a model 
            AE_surr = tf.keras.models.Model([latent_space, search_space, cost_space], ae_surr)
            
            # Add encoder network, and ensure the passed on design is equal to encoded one
            AES = tf.keras.models.Model([search_space, cost_space], AE_surr([Encoder(search_space), search_space, cost_space]))
        
        # Check if an adverasial network should be build
        if use_AAE:
            # Check if probability is a positive scalar
            if isinstance(ad_prob, collections.Sequence):
                raise TypeError("Discriminator weight is supposed to be a scalar")
            ad_prob = min(max(float(ad_prob), 0.0), 0.5)
            
            # Build the discriminator network (MLP)
            dis = tf.keras.layers.Dense(self.edim, activation='sigmoid')(latent_space)
            dis = tf.keras.layers.Dense(self.edim, activation='sigmoid')(dis)
            dis = tf.keras.layers.Dense(self.edim, activation='sigmoid')(dis)
            dis = tf.keras.layers.Dense(self.edim, activation='sigmoid')(dis)
            dis = tf.keras.layers.Dense(1, activation='sigmoid')(dis)
            
            Dis = tf.keras.models.Model(latent_space, dis)
            
            # Build an untrainable version of the discriminator
            Dis_const = tf.keras.models.Model(latent_space, Dis(latent_space))
            Dis_const.trainable = False
            Dis.trainable = True
            
            # Build an untrainable version of the encoder
            Encoder_const = tf.keras.models.Model(search_space, Encoder(search_space))
            Encoder_const.trainable = False
            Encoder.trainable = True
            
            # Build loss function L_D2, which trains encoder
            dis_e = Encoder(search_space)
            dis_e = Dis_const(dis_e)
            dis_e = tf.keras.layers.Lambda(lambda x: - K.log(1 - x))(dis_e)
            
            Dis_E = tf.keras.models.Model(search_space, dis_e)
            
            # Build loss function L_D1, which trains the discriminator, and takes encoded training samples
            # as well as randomly generate dlatent space samples as input
            dis_d = Encoder_const(search_space)
            dis_d = Dis(dis_d)
            dis_d = tf.keras.layers.Lambda(lambda x: - 0.5 * (K.log(x[0]) + K.log(1 - x[1])))([dis_d, Dis(latent_space)])
            
            Dis_D = tf.keras.models.Model([search_space, latent_space], dis_d)
                                                
        # Train only the Autoencoder    
        if not use_Surr and not use_AAE:
            AE.compile('Adam', 'mse')
            AE.fit(X_train, X_train, epochs = epochs, batch_size = batch_size, verbose = 2)
        
        # Train Autoenecoder with surrogate network
        elif use_Surr and not use_AAE:
            AES.compile('Adam', 'mae')
            AES.fit([X_train, C_train], np.zeros_like(C_train), \
                         epochs = epochs, batch_size = batch_size, verbose = 2)
        
        # Train Adversarial Autoencoder
        elif not use_Surr and use_AAE:
            AE.compile('Adam', 'mse')
            Dis_D.compile('Adam', 'mae')
            Dis_E.compile('Adam', 'mae')
            Dis_num_batches = int(np.ceil(len(X_train) / batch_size))
            for epoch in range(epochs):
                if mpi_rank == 0:
                    print('Epoch {}/{}'.format(epoch + 1, epochs))
                time_start = time.time()
                loss_AE = AE.fit(X_train, X_train, epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                # randomly train encoder and discriminator to fool each other
                prob_test=np.random.rand()
                if prob_test < ad_prob or epoch == 0:
                    loss_D = Dis_D.fit([X_train, np.random.rand(len(X_train), self.edim)], np.zeros((len(X_train), 1)), \
                                       epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                if prob_test > 1 - ad_prob or epoch == 0:                    
                    loss_E = Dis_E.fit(X_train,np.zeros((len(X_train),1)), \
                                       epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                time_end = time.time()
                if mpi_rank == 0:
                    print('{}/{} - {:0.0f}s - loss: {:0.4f} Rec, {:0.4f} D1, {:0.4f} D2'.format(Dis_num_batches, \
                                                                                            Dis_num_batches, \
                                                                                            time_end - time_start, \
                                                                                            loss_AE, loss_D, loss_E))
        
        # Train an Adversarial Autoencoder with surrogate network            
        else:
            AES.compile('Adam', 'mae')
            Dis_D.compile('Adam', 'mae')
            Dis_E.compile('Adam', 'mae')
            Dis_num_batches = int(np.ceil(len(X_train) / batch_size))
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch + 1, epochs))
                time_start = time.time()
                loss_AE = AES.fit([X_train, C_train], np.zeros_like(C_train), \
                                      epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                prob_test=np.random.rand()
                if prob_test < 0.5 * ad_prob or epoch == 0:
                    loss_D = Dis_D.fit([X_train, np.random.rand(len(X_train), self.edim)], np.zeros((len(X_train), 1)), \
                                       epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                if prob_test > 1 - ad_prob or epoch == 0:                    
                    loss_E = Dis_E.fit(X_train, np.zeros((len(X_train), 1)), \
                                       epochs = 1, batch_size = batch_size, verbose = 0).history['loss'][-1]
                time_end = time.time()
                if mpi_rank == 0:
                    print('{}/{} - {:0.0f}s - loss: {:0.4f} Rec, {:0.4f} D1, {:0.4f} D2'.format(Dis_num_batches, \
                                                                                                Dis_num_batches, \
                                                                                                time_end - time_start, \
                                                                                                loss_AE, loss_D, loss_E))
            
            
            
        # calculate final loss and save decoder internally
        try:
            Z_train = Encoder.predict(X_train, batch_size = batch_size)
        except:
            Z_train = np.zeros(len(X_train), self.edim)
            for i in range(len(X_train)):
                Z_train[[i],:] = Encoder.predict(X_train[[i],:])          
        
        try:
            X_train_pred = Decoder.predict(Z_train, batch_size = batch_size)
        except:
            X_train_pred = np.zeros(X_train.shape)
            for i in range(len(X_train)):
                X_train_pred[[i],:] = Decoder.predict(Z_train[[i],:])

        final_loss = np.mean((X_train - X_train_pred)**2)
        
        self.Decoder = Decoder
        
        return Encoder, Decoder, final_loss
    
    def add_decoder(self, Decoder):
        '''
        Allows the possibility to add an decoder before the third step of AEGO, 
        skipping the first two steps.

        Parameters
        ----------
        Decoder : keras.Model
            The decoder which maps samples from latent space to the search space.
        '''
        # Check if decoder is right input
        if not isinstance(Decoder, tf.keras.models.Model):
            raise TypeError("Decoder must be a keras model")
        # Check if decoder maps into the right design space
        if Decoder.output_shape[1:] != self.dim:
            raise ValueError("Decoder does not project into design_space")
        
        self.Decoder = Decoder
        self.edim = int(self.Decoder.input_shape[1])
        
    def DE(self, zmin = 0.0, zmax = 1.0, gamma = 100, G = 500, F = 0.6, chi_0 = 0.95, mu = 0, LO_params = None, 
           use_MPI = False, mpi_comm = None, mpi_rank = 0, mpi_size = 1):
        '''
        Allows for the gloabl optimization over latent space using differential evolution.
        This is the third step of the AEGO-algorithm

        Parameters
        ----------
        zmin : float, optional
            Minimal value sample elements can take. The default is 0.
        zmax : TYPE, optional
            Maximal value sample elements can take. The default is 1.
        gamma : int, optional
            Number of samples use for DE. The default is 100.
        G : int, optional
            Number of generations to use DE. The default is 500.
        F : float, optional
            Factor used for generating offspring. The default is 0.6.
        chi_0 : float, optional
            Probability of crossover. The default is 0.95.
        mu : int, optional
            Number of loacl optimization steps included in c_mu. The default is 0.
        LO_params : dictionary, optional
            Parameters that are used by the local optimizer, like for example Adam. Defualt is None.
        use_MPI : bool, optional
            Check if a mulitple processors should be used. The default is False
        mpi_comm : mpi4py.MPI.Intracomm, optional.
            Object allowing for communication between separate processes. The default is None.
        mpi_rank : int, optional
            ID of process used. The default is 0.
        mpi_size : int, optional
            Number of processes used. The default is 1.

        Returns
        -------
        Z_min : numpy.ndarray
            Optimized latent space samples.
        X_min : numpy.ndarray
            Optimized decoded latent space samples.
        C_min : numpy.ndarray
            History of cost function over optimization run.

        '''
        # Check if Decoder has been already assigned
        if not hasattr(self, 'Decoder'):
            raise AttributeError("Decoder has not yet been added to this object")
        # Check if zmin is a scalar
        if isinstance(zmin, collections.Sequence):
            raise TypeError("Minimum value is supposed to be a scalar")
        zmin = float(zmin)
        # Check if zmax is a scalar
        if isinstance(zmax, collections.Sequence):
            raise TypeError("Minimum value is supposed to be a scalar")
        zmax = float(zmax)
        # Check if gamma is a scalar
        if isinstance(gamma, collections.Sequence):
            raise TypeError("Number of samples is supposed to be a scalar")
        gamma = int(gamma)
        # Check if G is a scalar
        if isinstance(G, collections.Sequence):
            raise TypeError("Number of generations is supposed to be a scalar")
        G = int(G)
        # Check if F is a scalar
        if isinstance(F, collections.Sequence):
            raise TypeError("Offspring factor is supposed to be a scalar")
        F = float(F)
        # Check if chi_0 is a scalar
        if isinstance(chi_0, collections.Sequence):
            raise TypeError("Crossover probability is supposed to be a scalar")
        chi_0 = float(chi_0)
        # Check if mu is a scalar
        if isinstance(mu, collections.Sequence):
            raise TypeError("Number of LO steps is supposed to be a scalar")
        # Ensure mu is positive
        mu = max(int(mu), 0)
        
        # Determine the number of batches needed to run all samples
        num_batches_perrank = int(np.ceil(gamma / (self.max_num_samples * mpi_size)))
        # Calculate the interger number of samples in each batch
        num_batch_samples = int(np.floor( gamma / (num_batches_perrank * mpi_size)))
        # Calculate the number of samples that results from an integer number of batches
        # with an interger number of samples each
        gamma_perrank = num_batches_perrank * num_batch_samples
        if mpi_rank == 0:
            print('Actual number of samples used: {}/{}'.format(gamma_perrank * mpi_size, gamma))
        
        # Determine the samples belonging to each batch
        batch_index = np.arange(gamma_perrank).reshape(num_batches_perrank, num_batch_samples)
        
        # Set up the cost function with the spicific number of batch samples
        self.cost.prepare_LO(num_samples = num_batch_samples, **LO_params)
        
        if mpi_rank == 0:
            print('Differential Evolution - generation {}'.format(0))
        # Set up the initial random samples
        Z_de_rank = np.random.rand(gamma_perrank, self.edim)*(zmax - zmin) + zmin
        # calculate the initial cost function values
        X_de_rank = np.zeros([gamma_perrank] + list(self.dim))
        C_de_rank = np.zeros((G + 1, gamma_perrank))
        if self.num_constraints > 0:
            G_de_rank = np.zeros((G + 1, gamma_perrank, self.num_constraints))
        
        for i in range(num_batches_perrank):
            self.cost.LO_reset_state(**LO_params)
            X_de_rank[batch_index[i],:] = self.Decoder.predict(Z_de_rank[batch_index[i],:]) * self.x_interval + self.x_min_value
            if self.documentation == 'AEGO':
                X_de_rank[batch_index[i],:] = self.cost.LO(X_de_rank[batch_index[i],:], mu)
                if self.num_constraints > 0:
                    (C_de_rank[0,batch_index[i]],
                     G_de_rank[0,batch_index[i],:]) = self.cost.evaluate(X_de_rank[batch_index[i],:])
                else:
                    C_de_rank[0,batch_index[i]] = self.cost.evaluate(X_de_rank[batch_index[i],:])
            else:
                if self.num_constraints > 0:                    
                    (X_de_rank[batch_index[i],:], C_help, G_help) = self.cost.LO(X_de_rank[batch_index[i],:], mu)
                    C_de_rank[0,batch_index[i]] = C_help[-1,:]
                    G_de_rank[0,batch_index[i],:] = G_help[-1, :,:]
                else:
                    (X_de_rank[batch_index[i],:], C_help) = self.cost.LO(X_de_rank[batch_index[i],:], mu)
                    C_de_rank[0,batch_index[i]] = C_help[-1, :]
  
        
        # iterate over number of generations fro optimization
        for g in range(1,G + 1): 
            if mpi_rank == 0:
                print('Differential Evolution - generation {}'.format(g))
            # if more than one process is used, broadcast Z over all ranks
            test_case_rank = np.floor(np.random.rand(gamma_perrank,3) * (gamma_perrank * mpi_size - 1e-7)).astype('int')
            
            # check if information transfer is required
            if mpi_size > 1:
                # Collect data in mpi_rank == 0
                if mpi_rank == 0:
                    Z_de_receive = np.empty((mpi_size, gamma_perrank, self.edim))
                else:
                    Z_de_receive = None
                mpi_comm.Barrier()
                mpi_comm.Gather(Z_de_rank, Z_de_receive, root = 0)
                if mpi_rank == 0:
                    Z_de = Z_de_receive.reshape((gamma_perrank * mpi_size, self.edim))
                else:
                    Z_de = None
                # Broadcast data to rest of processes
                Z_de = mpi_comm.bcast(Z_de, root = 0)
                
                # Reproduction between differnt individuals from the population is perforemd
                # Determin eparent samples for each child                
                Za_rank = np.copy(Z_de[test_case_rank[:,0],:])
                Zb_rank = np.copy(Z_de[test_case_rank[:,1],:])
                Zc_rank = np.copy(Z_de[test_case_rank[:,2],:])                
            else:
                # Reproduction between differnt individuals from the population is perforemd
                # Determin eparent samples for each child
                Za_rank = np.copy(Z_de_rank[test_case_rank[:,0],:])
                Zb_rank = np.copy(Z_de_rank[test_case_rank[:,1],:])
                Zc_rank = np.copy(Z_de_rank[test_case_rank[:,2],:])
                
            # Combine parents to form child
            Z_com_rank = Za_rank + F * (Zb_rank - Zc_rank)
            
            ## Crossover between child and parent is performed
            # determine elements to cross
            crossover_rank = np.random.rand(gamma_perrank, self.edim) > chi_0
            # aply crossover
            Z_com_rank[crossover_rank] = np.copy(Z_de_rank[crossover_rank])
            
            ## Boundaries of latent space are enforced
            Z_com_rank[Z_com_rank < zmin] = zmin
            Z_com_rank[Z_com_rank > zmax] = zmax 
            
            ## Evaluate the children samples
            X_com_rank = np.zeros([gamma_perrank] + list(self.dim))
            for i in range(num_batches_perrank):
                self.cost.LO_reset_state(**LO_params)
                X_com_rank[batch_index[i],:] = self.Decoder.predict(Z_com_rank[batch_index[i],:])  * self.x_interval + self.x_min_value
                if self.documentation == 'AEGO':
                    X_com_rank[batch_index[i],:] = self.cost.LO(X_com_rank[batch_index[i],:], mu)
                    if self.num_constraints > 0:
                        (C_de_rank[g,batch_index[i]],
                         G_de_rank[g,batch_index[i],:]) = self.cost.evaluate(X_com_rank[batch_index[i],:])
                    else:
                        C_de_rank[g,batch_index[i]] = self.cost.evaluate(X_com_rank[batch_index[i],:])
                else:
                    if self.num_constraints > 0:                    
                        (X_com_rank[batch_index[i],:], C_help, G_help) = self.cost.LO(X_com_rank[batch_index[i],:], mu)
                        C_de_rank[g,batch_index[i]] = C_help[-1, :]
                        G_de_rank[g,batch_index[i],:] = G_help[-1, :,:]
                    else:
                        (X_com_rank[batch_index[i],:], C_help) = self.cost.LO(X_com_rank[batch_index[i],:], mu)
                        C_de_rank[g,batch_index[i]] = C_help[-1, :]
            
            # find the children which represent improvement
            replace = C_de_rank[g,:] <= C_de_rank[g-1,:]
            
            # keep best samples and replace unfit parents
            C_de_rank[g,np.invert(replace)] = C_de_rank[g-1, np.invert(replace)]
            if self.num_constraints > 0:
                G_de_rank[g,np.invert(replace),:] = G_de_rank[g-1, np.invert(replace),:]
            Z_de_rank[replace,:] = Z_com_rank[replace,:]
            X_de_rank[replace,:] = X_com_rank[replace,:]
        
        # return final samples, as well as the progress of cost function values
        if self.num_constraints > 0:
            return Z_de_rank, X_de_rank, C_de_rank, G_de_rank
        else:
            return Z_de_rank, X_de_rank, C_de_rank
                
    def Post_processing(self, X_post_rank, nu = 100, LO_params = None,
                        use_MPI = False, mpi_comm = None, mpi_rank = 0, mpi_size = 1):
        '''
        Allows for postprocessing of the decoed optimized latent space samples using Adam.
        This the fourth and final step of the AEGO algorithm.

        Parameters
        ----------
        Xmin : numpy.ndarray
            The decoded optimized samples to be post processed.
        LO_params : dictionary, optional
            Parameters that are used by the local optimizer, like for example Adam. Defualt is None.
        nu : int, optional
            Number of loacl optimization steps. The default is 100.
        use_MPI : bool, optional
            Check if a mulitple processors should be used. The default is False
        mpi_comm : mpi4py.MPI.Intracomm, optional.
            Object allowing for communication between separate processes. The default is None.
        mpi_rank : int, optional
            ID of process used. The default is 0.
        mpi_size : int, optional
            Number of processes used. The default is 1.

        Returns
        -------
        X_min : numpy.ndarray
            Post-processed optimized decoded latent space samples.
        C_min : numpy.ndarray
            History of cost function over post-processing run.

        '''
        # check if xmin is an array
        if not isinstance(X_post_rank, np.ndarray):
            raise TypeError("Input is supposed to be a numpy array")
        # Check if input dimensionality matches the design space
        if X_post_rank.shape[1:] != self.dim:
            raise TypeError("Input dimensions do not fit cost function")
        # Check if nu is an scalar
        if isinstance(nu, collections.Sequence):
            raise TypeError("Number of LO steps is supposed to be a scalar")
        # Ensure that nu is positive
        nu = max(int(nu),0)
        
        # Determine the number of sampless
        num_samples_perrank_old = len(X_post_rank)
        # Determine the number of batches needed to run all samples
        num_batches_perrank = int(np.ceil(num_samples_perrank_old / self.max_num_samples))
        # Calculate the interger number of samples in each batch
        num_batch_samples = int(np.floor( num_samples_perrank_old / num_batches_perrank))
        # Calculate the number of samples that results from an integer number of batches
        # with an interger number of samples each
        num_samples_perrank = num_batches_perrank * num_batch_samples
        # Determine the samples belonging to each batch
        batch_index = np.arange(num_samples_perrank).reshape(num_batches_perrank,num_batch_samples)
        
        if mpi_rank == 0:
            print('Actual number of samples used: {}/{}'
                  .format(num_samples_perrank * mpi_size, num_samples_perrank_old * mpi_size))
        
        # Set up the cost function with the spicific number of batch samples
        self.cost.prepare_LO(num_samples = num_batch_samples, **LO_params)
        
        if mpi_rank == 0:
            print('Post processing - iteration {}'.format(0))
        # calculate the initial cost function values
        C_post_rank=np.zeros((nu + 1, num_samples_perrank))
        if self.num_constraints > 0:
            G_post_rank=np.zeros((nu + 1, num_samples_perrank, self.num_constraints))
        
        
        if self.documentation == 'AEGO':
            for i in range(num_batches_perrank):
                if self.num_constraints > 0:
                    (C_post_rank[0,batch_index[i]], 
                     G_post_rank[0,batch_index[i],:]) = self.cost.evaluate(X_post_rank[batch_index[i],:])  
                else:
                    C_post_rank[0,batch_index[i]] = self.cost.evaluate(X_post_rank[batch_index[i],:])
               
                # Reset optimizer params if necessary (for example momentum)
                self.cost.LO_reset_state(**LO_params)  
                for g in range(1, nu + 1):
                    if mpi_rank == 0:
                        print('Generating samples - iteration {} - batch {}'.format(g,i))
                    X_post_rank[batch_index[i],:] = self.cost.LO(X_post_rank[batch_index[i],:], 1)
                    if self.num_constraints > 0:
                        (C_post_rank[g,batch_index[i]], 
                         G_post_rank[g,batch_index[i],:]) = self.cost.evaluate(X_post_rank[batch_index[i],:])  
                    else:
                        C_post_rank[g,batch_index[i]] = self.cost.evaluate(X_post_rank[batch_index[i],:])
        else:             
            # iterate for each step
            for i in range(num_batches_perrank):
                # Reset optimizer params if necessary (for example momentum)
                self.cost.LO_reset_state(**LO_params)            
                
                if mpi_rank == 0:
                    print('Generating samples - {} iterations - batch {}'.format(nu,i))                    
                if self.num_constraints > 0:
                    (X_post_rank[batch_index[i],:],
                     C_post_rank[:, batch_index[i]],
                     G_post_rank[:, batch_index[i],:]) = self.cost.LO(X_post_rank[batch_index[i],:], nu)
                else:
                    (X_post_rank[batch_index[i],:],
                     C_post_rank[:, batch_index[i]]) = self.cost
        
        if self.num_constraints > 0:
            return X_post_rank, C_post_rank, G_post_rank
        else:
            return X_post_rank, C_post_rank