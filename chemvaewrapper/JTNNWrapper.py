from dgllife.data import JTVAEDataset, JTVAECollator
from torch.utils.data import DataLoader
from dgllife.model import load_pretrained
import torch
import numpy as np
#from tqdm import tqdm
from tqdm.notebook import tqdm

class JTNNWapper:
    def __init__(self,
                 model=None):
        
        if model is None:
            self.model = load_pretrained('JTNN_ZINC') # Pretrained model loaded
        else:
            self.model=model
            
        self.model.eval()
        
        self.path_smiles = 'smiles.tmp'
            
    def _write_smiles(self,smiles_list):
        with open(self.path_smiles, mode='w') as f:
            f.write("\n".join(smiles_list))
            
    def _init_loader(self):
        dataset = JTVAEDataset(data=self.path_smiles, vocab=self.model.vocab, training=False)
        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=JTVAECollator(False),
           )
        
    def encode(self,smiles_list):
        self._write_smiles(smiles_list)
        self._init_loader()
        model=self.model
        v_list=[]
        
        #for it, batch in tqdm(enumerate(self.dataloader)):
        ite=iter(self.dataloader)

        for it in tqdm(range(len(self.dataloader))):
            try:
            #if True:
                batch=next(ite)
                gt_smiles = batch['mol_trees'][0].smiles

                _, tree_vec, mol_vec = model.encode(batch)

                mol_mean = model.G_mean(mol_vec)
                tree_mean = model.T_mean(tree_vec)

                # Following Mueller et al.
                tree_log_var = -torch.abs(model.T_var(tree_vec))
                epsilon = torch.randn(1, model.latent_size // 2)
                tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon

                mol_log_var = -torch.abs(model.G_var(mol_vec))
                epsilon = torch.randn(1, model.latent_size // 2)
                mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
                
                v1=tree_vec.detach().numpy().copy()
                v2=mol_vec.detach().numpy().copy()
                v=np.concatenate([v1,v2],axis=-1)
                v_list.append(v.reshape(-1))
            except:
                #print("error: ",gt_smiles)
                print("error")
                
                v_list.append(np.zeros(model.latent_size,dtype=np.float32))
                
        return np.array(v_list)
    
    
    def decode(self,vecs):
        half_size=int(self.model.latent_size/2)
        sm_list=[]
        for i,v in enumerate(vecs):
            tree_vec=v[:half_size].reshape(-1,half_size)
            mol_vec=v[half_size:].reshape(-1,half_size)

            tree_vec=torch.from_numpy(tree_vec.astype(np.float32)).clone()
            mol_vec=torch.from_numpy(mol_vec.astype(np.float32)).clone()
            
            try:
                dec_smiles = self.model.decode(tree_vec, mol_vec)
            except:
                print("error!", i)
                dec_smiles="Error"
                
            sm_list.append(dec_smiles)
        return sm_list