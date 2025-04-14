import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AbstractAF:
    """Abstract Activation Function.
    This is an abstract class used to represent
    activation functions in a MultiLayer Perceptitron.

    Each Activation function must have a name
    and implement the following three functions:
    - fw(w,x)         represents a forward pass through the MLP
    - bp_w(w,x)       represents a dL/dw backprop through the MLP
    - bp_x(w,x)       represents a dL/dh backprop through the MLP <- TODO look into this.....

    This class does not implement any of the three functions.
    Child-classes MUST implement all three functions for 
    backprop to work properly.

    In the current implementation, the following classes are the only valid subclasses:
    - LinearAF
    - ReluAF
    """
    def __init__(self):
        self.name = "Abstract"

    def __repr__(self):
        """Overwrites the representation with class name.
        This function makes the print look cleaner :) 
        """
        return f"<ActivationFunction:{self.name}>"
    
    def fw(self,w,x):
        raise NotImplementedError("Abstract Class cannot run functions.  Please use a subclass.")

    def bp_w(self,w,x):
        raise NotImplementedError("Abstract Class cannot run functions.  Please use a subclass.")

    def bp_x(self,w,x):
        raise NotImplementedError("Abstract Class cannot run functions.  Please use a subclass.")

class MeanSquaredErrorAF(AbstractAF):
    """Mean Squared Error function"""
    def __init__(self):
        super().__init__()
        self.name = "MSE"
        self.axis = 0

    def fw(self,f,y):
        return   np.mean((f-y)**2,axis=1).item()

    def bp(self,f,y):
        return 2*np.mean((f-y),   axis=self.axis)

class LinearAF(AbstractAF):
    """Linear Activation Function"""
    def __init__(self):
        super().__init__()
        self.name = "Linear"
    
    def fw(self,w,x):
        return w.T.dot(x)

    def bp_w(self,w,x):
        return x

    def bp_x(self,w,x):
        return w

class ReluAF(AbstractAF):
    """Relu Activation Function"""
    def __init__(self):
        super().__init__()
        self.name = "Relu"
        
    def fw(self,w,x):
        return np.maximum(0,w.T.dot(x))

    def bp_w(self,w,x):
        print("wtx:",(w.T.dot(x) > 0).shape)
        print("x:",x.shape,"(expected)")
        return x.dot((w.T.dot(x) > 0).T)

    def bp_x(self,w,x):
        print("wtx:",(w.T.dot(x) > 0).shape)
        print("w:",w.shape,"(expected)")
        return (w).dot(w.T.dot(x) > 0)
    

class MLP:
    """MultiLayer Perceptron
    Implementation Notes:
    - input and output layers must be defined explicitly.
    """
    def __init__(self,seed=None):
        np.random.seed(seed)
        self.layers  = []
        self.weights = [
            np.array([[ 0.13876319,  0.22640905, -0.2853099 ,  0.21473718,  0.15649755,
                        0.37585086, -0.32501255, -0.21397295, -0.01192419, -0.06718183,
                        0.01381539,  0.3829614 , -0.10261642, -0.18769519,  0.41081833,
                        -0.11705712,  0.1943435 , -0.37700067, -0.18396392, -0.40779517],
                        [ 0.10782415, -0.40525728,  0.0371851 ,  0.25192093, -0.15933668,
                        -0.22124293,  0.09782554,  0.39318228,  0.15874965, -0.1396638 ,
                        0.05661196,  0.44283865,  0.57519204, -0.35586828,  0.13351117,
                        0.06998226,  0.03503674, -0.54733498,  0.27167423, -0.11141699],
                        [ 0.63085993,  0.51946617, -0.2434313 ,  0.38954271,  0.07995638,
                            0.3476454 , -0.01709522,  0.6481928 ,  0.04939612,  0.4584661 ,
                            0.30407065,  0.39810144,  0.78771624, -0.2712758 ,  0.21225954,
                            0.58786751,  0.49476035, -0.09897648,  0.15936382,  0.39840242],
                        [ 0.48122506,  0.06890104,  0.78864858,  0.2338569 ,  0.57983993,
                            0.17018448,  0.49465241,  0.20026509,  0.25541868, -0.16190038,
                            0.19786829,  0.32276974,  0.3037967 ,  0.17278417,  0.37048908,
                            -0.11726633,  0.12248824, -0.08559875,  0.69561037,  0.50400982],
                        [-0.29249275,  0.10787834,  0.21775779, -0.18387509,  0.01344881,
                            -0.0206688 ,  0.51669386,  0.0415605 ,  0.56729631, -0.34637565,
                            -0.16237446, -0.39850441, -0.15616279,  0.07951577,  0.33800896,
                            -0.10688913,  0.42562239,  0.21619812,  0.13054705, -0.3171269 ],
                        [ 1.37523701,  1.81838566,  2.23628201,  1.9288745 ,  1.5030407 ,
                            1.28030918,  1.54424284,  2.35802976,  1.05689198,  1.97343708,
                            1.37011172,  2.04479809,  0.99747628,  2.40925681,  1.48475021,
                            2.14802472,  1.51309035,  2.39484028,  1.46388871,  1.47566823]]),
            np.array([[0.81575872, 0.78098074, 0.5713058 , 0.47660547, 0.63601814,
                            0.12536821, 0.90368448, 1.02623056, 0.20240518, 0.35937266],
                        [1.01565837, 0.26012786, 0.79631267, 0.47274152, 0.56843577,
                            1.05910281, 0.27598803, 0.82487326, 0.94333111, 0.37049239],
                        [0.31347593, 0.90033555, 1.02145695, 0.85670633, 1.09404761,
                            0.97376051, 1.12566749, 0.32697099, 1.12046413, 0.85004479],
                        [0.95591565, 0.91559007, 1.03161703, 0.57160382, 0.63477566,
                            0.821484  , 0.419323  , 0.62012458, 0.40777653, 0.99242701],
                        [0.92982766, 0.68124277, 0.62969887, 0.62228555, 0.503087  ,
                            1.04932008, 0.69812598, 0.36633577, 0.08703464, 0.13448399],
                        [0.68118702, 0.73531159, 0.62110877, 0.59169085, 0.96604779,
                            0.18163253, 0.12328816, 0.80501911, 0.4709547 , 0.95258051],
                        [0.80363419, 0.85096071, 0.70966519, 0.45811772, 0.74706579,
                            0.43665567, 0.67882086, 0.99824616, 0.33713184, 0.68945864],
                        [1.17976651, 0.54703092, 0.93012948, 0.52342921, 0.93021613,
                            1.12958236, 0.33222549, 1.13887426, 0.39451488, 0.25291673],
                        [0.91631221, 0.60158757, 0.4642562 , 0.20244916, 0.26014106,
                            0.54043355, 0.51670869, 0.26155518, 0.87553303, 0.37423621],
                        [0.94351464, 1.02882863, 0.88716365, 0.72079608, 0.41052383,
                            0.84466855, 0.99655714, 0.72064799, 0.83285145, 0.92958348],
                        [0.6178925 , 0.80548309, 0.5036223 , 0.89914767, 0.93457715,
                            0.7902878 , 0.40106725, 0.49179855, 1.0397315 , 0.07286859],
                        [0.51563558, 0.49993551, 0.57451453, 1.03775901, 0.24459861,
                            0.7814895 , 1.06803723, 1.03371611, 1.05336331, 0.37510127],
                        [0.99927267, 0.22603923, 0.38648731, 0.11680275, 0.60997321,
                            0.02661729, 0.10232599, 0.39835013, 0.77546679, 0.11187975],
                        [0.80508099, 0.82789822, 0.9139958 , 1.00151805, 0.74423743,
                            0.43024171, 1.17714617, 1.17071931, 0.74057901, 0.41662449],
                        [0.8661741 , 0.52155446, 0.98001828, 0.09927612, 0.73295715,
                            0.21281478, 0.34153784, 0.28717377, 0.95295181, 0.37815173],
                        [0.52185779, 0.94053123, 0.70929054, 0.55481425, 0.52702233,
                            0.98796436, 0.74734306, 0.82777455, 0.49343762, 0.14493528],
                        [0.23332507, 0.28444276, 0.75068527, 0.13578676, 0.50505335,
                            0.8457517 , 0.21280329, 0.42885102, 0.75565157, 0.38166656],
                        [1.12434282, 0.48839608, 1.33441102, 1.29295864, 0.71665596,
                            1.15030331, 1.17031254, 0.85608096, 0.42598152, 1.11133456],
                        [0.70242401, 0.27458019, 0.42551802, 0.80228575, 0.35478402,
                            0.30055115, 0.61780083, 0.3394756 , 0.16897147, 0.57782876],
                        [0.73906758, 0.95337549, 0.91401924, 0.96698846, 0.59424507,
                            0.89418329, 1.06394299, 0.28971739, 0.35013688, 0.53549198]]),
            np.array([[1.91845893],
                        [2.15922958],
                        [2.54854009],
                        [2.02603634],
                        [2.02847951],
                        [2.01256605],
                        [2.31830547],
                        [2.20053168],
                        [1.52545768],
                        [1.13347019]])]
        self.loss = MeanSquaredErrorAF()

    def add_layer(self,nodes:int,afunc:AbstractAF) -> None:
        """Adds a layer with a given number of nodes
        and a given Abstract Function"""
        self.layers.append(MLPLayer(nodes,afunc))

    def fw(self,x:np.array):
        # initialize x as the hidden value
        # of layer 0 (the input layer)
        self.layers[0].h = x

        # loop through and update x iteratively:
        for i in range(1,len(self.layers)):
            x = self.layers[i].fw(self.weights[i-1],x)

        # return x
        return x    


class MLPLayer:
    """Represents a single layer in the MLP.
    
    """
    def __init__(self,nodes,afunc):
        self.nodes = int(nodes)
        self.afunc = afunc
        self.h = None

    def __repr__(self):
        """overwrite representation for pretty print"""
        return "<MLPLayer: {nodes:"+f"{self.nodes},afunc:{self.afunc}"+"}>"

    def get_nodes(self):
        return self.nodes+0

    def fw(self,w:np.array,x:np.array):
        """store and return the
        post-activation values 
        of a forward pass."""
        self.h = self.afunc.fw(w=w,x=x)
        return self.h.copy()

    def bp_w(self,w:np.array):
        return self.afunc.bp_w(w=w,x=self.h)

    def bp_x(self,w:np.array):
        return self.afunc.bp_x(w=w,x=self.h)
