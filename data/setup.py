import numpy as np

def setup():
  N = type(None)
  V = np.array
  ARRAY = np.ndarray
  ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
  VS = Union[Tuple[V, ...], List[V]]
  VN = Union[V, N]
  VNS = Union[VS, N]
  T = torch.Tensor
  TS = Union[Tuple[T, ...], List[T]]
  TN = Optional[T]
  TNS = Union[Tuple[TN, ...], List[TN]]
  TSN = Optional[TS]
  TA = Union[T, ARRAY]


  D = torch.device
  CPU = torch.device('cpu')


  def get_device(device_id: int) -> D:
      if not torch.cuda.is_available():
          return CPU
      device_id = min(torch.cuda.device_count() - 1, device_id)
      return torch.device(f'cuda:{device_id}')


  CUDA = get_device

  current_directory = os.getcwd()
  save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
  os.makedirs(save_path, exist_ok=True)
  model_path = os.path.join(save_path, 'model_wieghts.pt')