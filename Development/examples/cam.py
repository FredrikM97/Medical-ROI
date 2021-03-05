from neural_network.utils.cam import get_cam
from neural_network.utils.cam_plots import plot_CAM_grid
from neural_network.utils import move_to_device, to_cpu_numpy
from neural_network.utils import interactive_slices, interactive_slices_masked
def cam_example(
        agent, 
        extractor_name:str='SmoothGradCAMpp',
        img_size:tuple=(79,224,224), 
        plot_type:str='grid',
        cmap:str='jet', 
        alpha:float=0.3, 
        observed_class:str=None, 
        load_image:str='CN'
    ):
    
    target_layer = 'model.layer4'
    model = type(agent.model.model).__name__
    
    image = {
        'CN':nib.load('data/SPM_categorised/AIH/CN/CN_ADNI_998.nii').get_fdata,
        'MCI':nib.load('data/SPM_categorised/AIH/MCI/MCI_ADNI_1586.nii').get_fdata,
        'AD':nib.load('data/SPM_categorised/AIH/AD/AD_ADNI_2975.nii').get_fdata
    }[load_image]()
    label_to_class = {
        0:'CN',
        1:'MCI',
        2:'AD'
    }
    class_to_label = {v: k for k, v in label_to_class.items()}
    
    image = torch.from_numpy(resize(image, img_size)).float()
    model = agent.model

    model, image = move_to_device(model, image, 'cuda')
    
    mask, predicted_label = get_cam(model, image, extractor_name=extractor_name, target_layer=target_layer, observed_class=class_to_label[observed_class])
    
    observed_class = observed_class if observed_class else load_image
    
    if plot_type == 'grid':
        fig = plot_CAM_grid(
            to_cpu_numpy(image), 
            mask,layer=target_layer, 
            label=label_to_class[predicted_label], 
            observed_class=observed_class,
            extractor=extractor_name,cmap=cmap, 
            alpha=alpha
        )
        
    elif plot_type == 'slice':
        testplot = interactive_slices()
        testplot.multi_slice_viewer(to_cpu_numpy(image))
        #testplot.cycle(0.1)
        testplot.close()
    elif plot_type == 'slice_masked':
        testplot = interactive_slices_masked()
        testplot.multi_slice_viewer(to_cpu_numpy(image), mask)
        #testplot.cycle(1)
        
        testplot.close()
    return fig
        