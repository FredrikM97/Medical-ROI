from typing import Optional, List, Tuple
from . import utils

class TSNE:
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1,79,224,224),
    ) -> None:

        # Obtain a mapping from module name to module instance for each layer in the model
        self.submodule_dict = dict(model.named_modules())

        if target_layer != None, 
            raise ValueError("A target layer must be selected!")
            
        if target_layer not in self.submodule_dict.keys():
            raise ValueError(f"Unable to find submodule {target_layer} in the model")
            
        self.target_layer = target_layer
        self.model = model
        # Init hooks
        self.hook_a: Optional[Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Forward hook
        self.hook_handles.append(self.submodule_dict[target_layer].register_forward_hook(self._hook_a))
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def clear_hooks(self) -> None:
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:

        output = self.model(images)
        current_outputs = output.cpu().numpy()
        features = np.concatenate((outputs, current_outputs))
        return features
    
    def _plot(self):
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            # convert the class color to matplotlib format
            color = np.array(colors_per_class[label], dtype=np.float) / 255
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label)

    def _precheck(self, class_idx: int, scores: Optional[Tensor] = None) -> None:
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if not isinstance(self.hook_a, Tensor):
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_tsne(class_idx, scores, normalized)

    def compute_tsne(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class
        Args:
            class_idx (int): output class index of the target class whose CAM will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the hooked model
            normalized (bool, optional): whether the CAM should be normalized
        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """
        
        # Get map weight & unsqueeze it
        weights = self._get_weights(class_idx, scores)
        weights = weights[(...,) + (None,) * (self.hook_a.ndim - 2)]  # type: ignore[operator, union-attr]

        return batch_cams

    def extra_repr(self) -> str:
        return f"target_layer='{self.target_layer}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"