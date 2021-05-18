from src.utils.plot import imshow
def calculate_average_cam(_cam_extractor, _image, classes=[0,1,2],n=10):
    masks = []
    for i in range(n):
        _class_scores, _class_idx = _cam_extractor.evaluate(_image)
        grid_image, grid_mask = _cam_extractor.grid_class(_class_scores, classes, _image,pad_value=0.5, max_num_slices=16, nrow=4)
        
        imshow(grid_mask)
        masks.append(grid_mask)
    return masks