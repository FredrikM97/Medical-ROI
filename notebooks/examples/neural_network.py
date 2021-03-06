
def train_model(config='resnet50'):
    agent = Agent(config)
    agent.fit()
    
def load_model(config='resnet50'):
    agent = Agent(config)
    agent.load_model()
