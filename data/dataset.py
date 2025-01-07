class TransformerDataset:
    def __init__(self, src_data, tgt_data, batch_size, pad_idx=0):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.pad_idx = pad_idx
        self.n_samples = len(src_data)
        
    def __len__(self):
        return self.n_samples // self.batch_size
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        src_batch = self.src_data[start_idx:end_idx]
        tgt_batch = self.tgt_data[start_idx:end_idx]
        
        tgt_y = tgt_batch[:, 1:]
        tgt_input = tgt_batch[:, :-1]
        
        return src_batch, tgt_input, tgt_y
