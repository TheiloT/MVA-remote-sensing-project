import torch
import os
from collections import OrderedDict



def adapt_sen1floods11_checkpoint_to_backbone_only_checkpoint(sen1floods11_prithvi_path="backbones/prithvi_sen1floods11"):
    # Adapt the checkpoint to the backbone only
    # This is necessary for the program to successfully read the checkpoint when training starts

    # Load the checkpoint
    checkpoint = torch.load(os.path.join(sen1floods11_prithvi_path, "sen1floods11_Prithvi_100M.pth"))

    # Remove the keys that are not related to the backbone
    new_checkpoint = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            new_checkpoint[new_key] = value

    # Rename old checkpoint
    os.rename(
        os.path.join(sen1floods11_prithvi_path, "sen1floods11_Prithvi_100M.pth"),
        os.path.join(sen1floods11_prithvi_path, "sen1floods11_Prithvi_100M_old.pth")
    )

    # Save the adapted checkpoint
    adapted_checkpoint_path = os.path.join(sen1floods11_prithvi_path, "sen1floods11_Prithvi_100M.pth")
    torch.save(new_checkpoint, adapted_checkpoint_path)

    print(f'Adapted checkpoint saved to {adapted_checkpoint_path}')
    
    
if __name__ == "__main__":
    adapt_sen1floods11_checkpoint_to_backbone_only_checkpoint()