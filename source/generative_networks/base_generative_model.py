class GenerativeModel(object):
    def __init__(self, params, **kwargs):
        """
        Initialization method for the GenerativeModel class

        Args:
            params (namespace object): contains the parameters used to initialize the class
        MJT Note: do we need the **kwargs argument?
        """
        self.params = params

    def prepare_population(self, population):
        return population
    
    def optimize(self):
        """
        optimize function not implemented in super class
        """
        raise NotImplementedError