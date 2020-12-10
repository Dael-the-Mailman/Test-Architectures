class DuelDDQNet:
    name = 'DuelDDQNet'
    def __init__(self):
        pass

    def forward(self):
        pass
    
    def save_checkpoint(self):
        print('Saving Checkpoint...')
        T.save(self.state_dict(), './checkpoints/DuelDDQNet')
    
    def load_checkpoint(self):
        print('Loading Checkpoint...')
        self.load_state_dict(T.load('./checkpoints/DuelDDQNet'))