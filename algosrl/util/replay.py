from typing import List, Dict, Union
import numpy as np

from algos import AbstractParametered
# from algos import Observer

FTYPE = np.float32
DTYPE = np.int32

class ReplayBuffer(AbstractParametered):
    """
    A purely pythonic with numpy implementation of a replay buffer.

    A numpy ndarray is used as a FIFO queue. Gets around slow writes by using a 
    head index to identify the most recent additions. The advantage of implementing 
    this way over a dequeue or similar is that it also implements accessing experiences
    via indices. This is useful if an algorithm requires access to experiences from a given
    rollout/epoch.

    An additional advantage is that all ints and floats are stored as DTYPE and FTYPE constants
    respectively. The default is np.int32 and np.float32. The advantage of this is that
    all datatypes stored in the replaybuffer are consistent.  
    """
    def __init__(self, maxlen: int=30000, keys: List[str] = []):
        self.keys = keys
        #define this with the first experience
        self._dtype = None
        self._data = None
        self.maxlen = maxlen
        self._head_index = -1
        self._len = 0
        self._observe_bounds = False
        self._observed_min = None
        self._observed_max = None

    def add_key(self, key):
        """
        Adds key to replay buffer. This would be harmful if 
        collect experience has been called. (Not entirely sure of
        my intention when I wrote this function) 
        """
        if isinstance(key, str):
            self.keys.append(key)
        #I have nfi what the intention is here??? Column names are not string I guess.
        # else:
        #     self.keys += key
        else:
            raise ValueError(f'{key} is not a string')

    def collect_experience(self, exp:Dict[str, Union[FTYPE,DTYPE]], **kwargs):
        """
        Adds an entry to the ReplayBuffer using the states in 
        the observers with the same name as the keys that the 
        replay holds.

        If it is the first experience collected determine the dtype
        of each keys value and initiate the data array.
        """
        # if kwargs:
        #     exp = kwargs.copy()
        # else:
        #     exp = {key: Observer(key).state for key in self.keys}
        if self._dtype is None:
            self.generate_dtype(exp)
            self._data = np.zeros(shape=self.maxlen, dtype=self._dtype)
        self.append(exp)

    def generate_dtype(self, exp: Dict[str, Union[FTYPE,DTYPE]]):
        """Generates the dtype array to create np.ndarray with named columns


        :param exp: The experience dictionary
        :type exp: Dict[str, Union[FTYPE,DTYPE]]
        """        
        dtypes = []
        add_keys = self.keys == []
        for key, value in exp.items():
            dt = self.get_dtype(value)
            if add_keys:
                self.keys.append(key)
            if hasattr(value, 'dtype'): #numpy type
                dtype = (key, dt, value.shape)
            else: #non numpy type
                dtype = (key, dt)
            dtypes.append(dtype)
        self._dtype = np.dtype(dtypes)

    def get_dtype(self, value: Union[float, int])->np.dtype:
        """Determine the dtype of value

        :param value: the value of the experience
        :type value: Union[float, int]
        :raises ValueError: if the type of the value is not considered by the replay buffer
        :return: FTYPE or DTYPE depending if float or int
        :rtype: np.dtype
        """        
        if isinstance(value, np.ndarray):
            return self._determine_nd_array(value)
        elif isinstance(value,
                        (np.float_, np.float16, np.float32, np.float64, float)):
            return FTYPE
        elif isinstance(value, (np.int_, np.int16, np.int32, np.int64, int)):
            return DTYPE
        else:
            raise ValueError(f'dtype: {type(value)} not accounted for')

    def _determine_nd_array(self, value: np.ndarray)->np.dtype:
        """Determine dtype of ndarray

        :param value: The array to process
        :type value: np.ndarray
        :return: FTYPE or DTYPE
        :rtype: np.dtype
        """            
        if value.dtype in (np.float_, np.float16, np.float32, np.float32):
            return FTYPE
        elif value.dtype in (np.int_, np.int16, np.int32, np.int64):
            return DTYPE
        else:
            raise ValueError(f'dtype: {value.dtype} not accounted for')

    def append(self, exp: Dict[str, Union[FTYPE,DTYPE]])->None:
        """Add an experience to the ReplayBuffer

        :param exp: The experience to be added
        :type exp: Dict[str, Union[FTYPE,DTYPE]]
        """
        #Determine bounds      
        self.observe_bounds(exp)
        exp = np.array([tuple([value for _, value in exp.items()])],
                       dtype=self._dtype)
        assert len(exp.shape) == 1, 'Unexpected input conversion'
        self._head_index = (self._head_index + 1) % self.maxlen
        self._data[self._head_index] = exp
        if self._len < self.maxlen:
            self._len += 1

    def observe_bounds(self, exp:Dict[str, Union[FTYPE,DTYPE]])->None:
        """Record the upper and lower bounds seen by the replay buffer over the course of the experiment.

        :param exp: The experience
        :type exp: Dict[str, Union[FTYPE,DTYPE]]
        """        
        #don't observe
        if not self._observe_bounds: return
        #initialise at the first experience
        if self._observed_min is None:
            self._observed_min = self._observed_max = np.array(
                [tuple([value for _, value in exp.items()])],
                dtype=self._dtype)
        #modify as required (this may be faster/more practical to do every x episodes...)
        #less for loops
        else:
            for key, _ in exp.items():
                new_e = exp[key]
                if not hasattr(new_e, 'shape'):
                    new_e = np.array(new_e)
                new_e = new_e.reshape(self._observed_min[key].shape)
                upper_mask = new_e > self._observed_max[key]
                if np.sum(upper_mask):
                    self._observed_max[key][upper_mask] = new_e[upper_mask]
                lower_mask = new_e < self._observed_min[key]
                if lower_mask.sum():
                    self._observed_min[key][lower_mask] = new_e[lower_mask]

    def sample(self, batch_size: int)->np.ndarray:
        """Return batch_size random samples from the replay buffer 

        :param batch_size: The batch size
        :type batch_size: int
        :return: the batch of samples
        :rtype: np.ndarray
        """        
        if len(self) < batch_size:
            return self._data[np.random.randint(len(self),
                                                     size=len(self))]
        else:
            return self._data[np.random.randint(len(self),
                                                     size=batch_size)]

    def __getitem__(self, indices:Union[int,slice,list,np.ndarray,str,tuple])->np.ndarray:
        """Allows the class to be indexed based on indices. This does the logic required to transform the requested indices into 
        the actual indices of the buffer based on the head index. This is all done to prevent memory rewrites which lead to a significant
        speed up of the buffer/

        :param indices: The indices, 0 being the newest experience, len-1 being the oldest
        :type indices: Union[int,slice,list,np.ndarray,str]
        :raises IndexError: Attempted to slice with too many dimensions
        :return: The experiences at the coresponding indices
        :rtype: np.ndarray
        """        
        #single index
        indices = self._process_indices(indices)
        if not isinstance(indices, tuple):
            return self._data[indices]
        elif len(indices) == 2:
            return self._data[indices[0]][indices[1]]
        else:
            raise IndexError(
                f'{indices} with length {len(indices)} is too many indices for array'
            )

    def _process_indices(self, indices:Union[int,slice,list,np.ndarray,str,tuple]) -> tuple:
        """Gives support for indexing in the same way as numpy

        :param indices: the indices to process
        :type indices: Union[int,slice,list,np.ndarray,str,tuple]
        :return: 
        :rtype: tuple
        """        
        if not isinstance(indices, tuple):
            return self._process_index(indices)
        return tuple([self._process_index(val) for val in indices])

    def _process_index(self, index: Union[int, slice, list, np.ndarray, str])->Union[int, slice, np.ndarray]:
        """Allows the array to be indexed based on a variety of indices (like numpy arrays) and transforms these to the 
        circular buffer

        :param index: the index to process
        :type index: Union[int, slice, list, np.ndarray, str]
        :raises IndexError: The index is not of the correct type
        :return: The index converted to the circular buffer index
        :rtype: Union[int, slice, np.ndarray]
        """        
        if isinstance(index, int):
            return self._process_int_ind(index)
        elif isinstance(index, slice):
            return self._process_slice(index)
        elif isinstance(index, (list, np.ndarray)):
            return self._process_ind_array(index)
        elif isinstance(index, str):
            return index
        else:
            raise IndexError(f'{type(index)} is not a support index type')

    def _process_ind_array(self, ind_array: Union[list, np.ndarray])->np.ndarray:
        """Process an array of indices

        :param ind_array: An array of indices
        :type ind_array: Union[list, np.ndarray]
        :return: the transformed array
        :rtype: np.ndarray
        """        
        if isinstance(ind_array, list): ind_array = np.array(ind_array)
        ind_array[ind_array < 0] += len(self)
        return (ind_array + self._head_index + 1) % len(self)

    def _process_int_ind(self, ind: int)->int:
        """Process an int index

        :param ind: the index to transform
        :type ind: int
        :return: the transformed index
        :rtype: int
        """        
        if ind is None: return None
        if ind < 0: ind += len(self)
        return (ind + self._head_index + 1) % len(self)

    def _process_slice(self, slce: slice)->slice:
        """Process a slice

        :param slce: The slice to be transformed
        :type slce: slice
        :return: The transformed slice
        :rtype: slice
        """        
        start = self._process_int_ind(slce.start)
        stop = self._process_int_ind(slce.stop)
        step = self._process_int_ind(slce.step)
        return slice(start, stop, step)

    def datainds2normalinds(self, inds: np.ndarray)->np.ndarray:
        """Converts the circular buffer indices to typical list indices

        :param inds: The buffer inds
        :type inds: np.ndarray
        :return: the transformed inds
        :rtype: np.ndarray
        """        
        lessinds = inds <= self._head_index
        moreinds = inds > self._head_index
        inds[lessinds] += (len(self) - 1 - self._head_index)
        inds[moreinds] -= (self._head_index + 1)
        return inds

    def __len__(self)->int:
        """The length of the buffer

        :return: the length of the buffer
        :rtype: int
        """        
        return self._len

    def reset(self):
        """Remove all of the values in the buffer but otherwise remain the same.
        """        
        self._data = np.zeros(shape=self.maxlen, dtype=self._dtype)
        self._len = 0

    @classmethod
    def set_up_hyperparameters(cls):
        cls.hyperparameters['maxlen'].bounds = (10000, 50000)