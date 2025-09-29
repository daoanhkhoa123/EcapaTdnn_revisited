import torch

def save_parameters(self, path):
		torch.save(self.state_dict(), path)

def load_parameters(self, path):
    self_state = self.state_dict()
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "").replace("speaker_encoder.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)