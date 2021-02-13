class ActivationMapHook():
    
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.hook = None
        self.features = None
        
    def register(self):
        self.hook = self.module.register_forward_hook(self.callback)
        
    def unregister(self):
        self.hook.remove()
        
    def callback(self, module, input, output):
        self.features = output.detach().cpu().data.numpy()