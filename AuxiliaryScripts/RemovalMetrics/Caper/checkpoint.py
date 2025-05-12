import os
import torch

def save_checkpoint(epoch, step, model, optimizer, filename):
    state = {   'epoch':epoch,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }


    filename = filename.split('/logits')

    path = os.path.join('results/', filename[0])

    # Ensure directory exists
    if not os.path.exists(path):
        os.makedirs(path)   

    path = os.path.join(path, filename[1])

    torch.save(state, path)
   

 
def load_checkpoint(name):
    checkpoint = torch.load(os.path.join('results',name))
    if 'pth' in name:
        return checkpoint

    return checkpoint["state_dict"]
