class MyCustomDataset(Dataset):
    def __init__(self, config):
        # load all nii handle in a list
        self.files = os.listdir(path)
        #self.images_list = [nib.load(image_path) for image_path in path]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        nii_image = self.images_list[idx]
        data = torch.from_numpy(np.asarray(nii_image.dataobj))
        
        self.transform(img)
        return data, target
    
    transform = T.Compose([
        T.ToPILImage(),
        #T.CenterCrop(0.75 * 64),
        #T.Resize(image_size),
        T.ToTensor()])
    
def load_data():
    batch_size = 64
    transformed_dataset = vaporwaveDataset(ims=X_train)
    train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break