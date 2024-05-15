import h5py
import numpy as np
from torch.utils.data import Dataset

#Original
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


#Original
class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDataset_3D(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset_3D, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # Handle boundary case; use the same frame as previous if it's the first frame
            if idx == 0:
                prev_idx = 0
            else:
                prev_idx = idx - 1

            current_lr = np.expand_dims(f['lr'][idx] / 255., 0)
            prev_lr = np.expand_dims(f['lr'][prev_idx] / 255., 0)
            
            # Stack current and previous LR images along the channel dimension to create a 2-channel input
            input_lr = np.concatenate((current_lr, prev_lr), axis=0)  # Concatenating along the channel axis
            
            label_hr = np.expand_dims(f['hr'][idx] / 255., 0)

            return input_lr, label_hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
        

class EvalDataset_3D(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset_3D, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # Handle boundary case for the first frame
            if idx == 0:
                prev_idx = 0
            else:
                prev_idx = idx - 1
            # prev_idx = idx

            # Load the current and previous LR frames and scale them
            current_lr = np.expand_dims(f['lr'][str(idx)][:,:] / 255., 0)
            prev_lr = np.expand_dims(f['lr'][str(prev_idx)][:,:] / 255., 0)

            # Concatenate along the channel dimension to form a 2-channel input
            input_lr = np.concatenate((current_lr, prev_lr), axis=0)

            # Load and scale the HR frame
            label_hr = np.expand_dims(f['hr'][str(idx)][:,:] / 255., 0)

            return input_lr, label_hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
        


# class EvalDataset_3D(Dataset):
#     def __init__(self, h5_file):
#         super(EvalDataset_3D, self).__init__()
#         self.h5_file = h5_file

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_file, 'r') as f:
#             current_lr = np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0)
#             label_hr = np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)
#             current_plus_prior = np.concatenate((current_lr, current_lr), axis=0)
#             return current_plus_prior, label_hr 

#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f['lr'])


class TrainDataset_3D_multifile(Dataset):
    def __init__(self, h5_files):
        super(TrainDataset_3D_multifile, self).__init__()
        self.h5_files = h5_files  # List of h5 file paths
        self.indexes = self._prepare_indexes()

    def _prepare_indexes(self):
        # Prepare an index list where each entry is (file_index, index_in_file)
        indexes = []
        for file_index, file_path in enumerate(self.h5_files):
            with h5py.File(file_path, 'r') as f:
                num_frames = len(f['lr'])  # Assuming 'lr' dataset indicates the number of frames
                for idx in range(num_frames):
                    indexes.append((file_index, idx))
        return indexes

    # def __getitem__(self, idx):
    #     file_index, index_in_file = self.indexes[idx]
    #     with h5py.File(self.h5_files[file_index], 'r') as f:
    #         # Assuming each dataset has the same structure as before
    #         current_lr = np.expand_dims(f['lr'][str(index_in_file)][:] / 255., 0)
    #         label_hr = np.expand_dims(f['hr'][str(index_in_file)][:] / 255., 0)
    #         return current_lr, label_hr
    def __getitem__(self, idx):
        file_index, index_in_file = self.indexes[idx]
        with h5py.File(self.h5_files[file_index], 'r') as f:
            # Correct access: assuming each dataset only contains a simple array
            # Note: Using string keys like 'lr' and integer indexing for frames
            current_lr = np.expand_dims(f['lr'][index_in_file][:] / 255., 0)
            label_hr = np.expand_dims(f['hr'][index_in_file][:] / 255., 0)
            return current_lr, label_hr

    def __len__(self):
        return len(self.indexes)