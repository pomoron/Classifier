import timm, torch, os, shutil
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np

def get_embedding(image_list, batch_size, device, model_id='vit_large_patch16_dinov3.lvd1689m', embedding_size=1024):
    all_embeddings = torch.empty((0, embedding_size)).to(device)
    model = timm.create_model(
        model_id,
        pretrained=True,
        # features_only=True,
    ).to(device)
    model = model.eval()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # outputs
    for i in range(0, len(image_list), batch_size):
        batch = torch.stack([transforms(Image.open(img_path).convert('RGB')) for img_path in image_list[i:i+batch_size]]).to(device)    # Apply inference preprocessing transforms
        with torch.no_grad():
            output = model.forward_features(batch)   # output is unpooled, a (N, 261, 1024) shaped tensor
            embeddings = model.forward_head(output, pre_logits=True)    # output is a (N, num_features) shaped tensor
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize across the embedding dimensions
        all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    
    return all_embeddings

def cluster(em_np, eps=0.5, min_samples=5):
    em_clust = DBSCAN(eps=eps, min_samples=min_samples).fit(em_np)
    em_labels = em_clust.labels_
    em_unique_labels = set(em_labels)
    if -1 in em_unique_labels:
        em_unique_labels.remove(-1)  # remove noise label
    return em_labels, em_unique_labels

def largest_images(image_names, n=10, move_file=False):
    sizes_and_names = [(img.size[0] * img.size[1], path) for path, img in [(path, Image.open(path)) for path in image_names]]
    top_n = sorted(sizes_and_names, key=lambda x: x[0], reverse=True)[:n]   # Sort by area descending and take top n
    image_names = [name for _, name in top_n]     # Extract filenames

    if move_file:
        old_dir = os.path.dirname(image_names[0])
        dest_dir = os.path.join(old_dir, f'max_{old_dir.split("/")[-1]}')
        os.makedirs(dest_dir, exist_ok=True)
        for name in image_names:
            shutil.copy(name, os.path.join(dest_dir, os.path.basename(name)))
        image_names = [os.path.join(dest_dir, os.path.basename(name)) for name in image_names]
    return image_names