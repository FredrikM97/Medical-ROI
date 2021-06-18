__all__ = ['ActivationMapHook','SaveFeaturesHook']

class ActivationMapHook():
    """ """
    
    def __init__(self, module, name):
        """

        Parameters
        ----------
        module :
            
        name :
            

        Returns
        -------

        
        """
        self.module = module
        self.name = name
        self.hook = None
        self.features = None
        
    def register(self):
        """ """
        self.hook = self.module.register_forward_hook(self.callback)
        
    def unregister(self):
        """ """
        self.hook.remove()
        
    def callback(self, module, input, output):
        """

        Parameters
        ----------
        module :
            
        input :
            
        output :
            

        Returns
        -------

        
        """
        self.features = output.detach().cpu().data.numpy()
        #self.features = output.cpu().data.numpy()
        
class SaveFeaturesHook():
    """ """
    features=None
    
    def __init__(self, m): 
        """

        Parameters
        ----------
        m :
            

        Returns
        -------

        
        """
        self.hook = m.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output): 
        """

        Parameters
        ----------
        module :
            
        input :
            
        output :
            

        Returns
        -------

        
        """
        self.features = ((output.cpu()).data).numpy()
        
    def remove(self): 
        """ """
        self.hook.remove()
        