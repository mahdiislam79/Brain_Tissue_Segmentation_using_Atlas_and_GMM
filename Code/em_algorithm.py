import numpy as np
from sklearn.cluster import KMeans
import nibabel as nib
from scipy.stats import multivariate_normal


# def initialize_clusters(data, n_clusters=3, random_state=0):
#     """Initialize clusters using k-means for mu, sigma, and pi."""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
#     mu = kmeans.cluster_centers_
#     labels = kmeans.labels_
#     sigma = np.array([np.std(data[labels == i], axis=0) for i in range(n_clusters)])
#     pi = np.ones(n_clusters) / n_clusters  # Equal probability for each cluster initially
#     return mu, sigma, pi

def normalize_image(image, new_min=1, new_max=255):
    """
    Normalize the image intensities to a specified range.
    
    Args:
        image (numpy.ndarray): Original image array.
        new_min (int): New minimum value for normalization (default: 1).
        new_max (int): New maximum value for normalization (default: 255).
    
    Returns:
        numpy.ndarray: Normalized image in the specified range.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized_image.astype(np.uint8)


# def em_preprocessing(image_path, mask_path):
#     """
#     Load and normalize an image, then extract data based on a ground truth mask.

#     Args:
#         image_path (str): Path to the image file (e.g., T1 or FLAIR).
#         mask_path (str): Path to the ground truth mask file.

#     Returns:
#         numpy.ndarray: Extracted and normalized data from the image where the mask is greater than 0.
#     """
#     # Load the image and mask
#     image = nib.load(image_path)
#     mask = nib.load(mask_path)

#     # Get image data
#     image_data = image.get_fdata()

#     # Normalize the image
#     normalized_image_data = (image_data - np.mean(image_data)) / np.std(image_data)

#     # Get mask data
#     mask_data = mask.get_fdata()

#     # Extract data where mask is greater than 0
#     extracted_data = normalized_image_data[mask_data > 0]

#     return extracted_data

# def em_post_processing(data_1d, gt_data, mu_em, responsibilities, num_classes=3):
#     """
#     Post-process the results of the EM algorithm to map labels consistently and reconstruct a 3D label map.

#     Args:
#         data_1d (numpy.ndarray): 1D array of data used in the EM algorithm.
#         gt_data (numpy.ndarray): Ground truth mask used for extracting valid regions.
#         mu_em (numpy.ndarray): Mean values for each cluster from the EM algorithm.
#         responsibilities (numpy.ndarray): Responsibilities matrix from the EM algorithm.
#         num_classes (int): Number of clusters (default: 3).

#     Returns:
#         numpy.ndarray: 3D label map with consistent labels.
#     """
#     # Assign labels based on maximum responsibility
#     labels = np.argmax(responsibilities, axis=1)

#     # Ensure consistent label mapping by sorting clusters based on their means
#     sorted_indices = np.argsort(mu_em[:, 0])  # Sort by mean intensities
#     label_mapping = {sorted_indices[i]: i + 1 for i in range(num_classes)}  # Map to 1, 2, 3, ...
    
#     # Map the labels to consistent labels
#     mapped_labels = np.vectorize(label_mapping.get)(labels)

#     # Create a 3D label map initialized to zero
#     label3d = np.zeros_like(gt_data)

#     # Assign the mapped labels to the corresponding locations in the 3D label map
#     label3d[gt_data > 0] = mapped_labels

#     return label3d

def compute_mu_sigma_pi_hard_labels(image, tissue_model):
    """
    Compute mu (mean), sigma (standard deviation), and pi (cluster probabilities) using hard-labeled intensities.

    Args:
        image (numpy.ndarray): 3D image array with normalized voxel intensities in range 1-255.
        tissue_model (pd.DataFrame): Tissue model loaded from a CSV with probabilities for each tissue type.

    Returns:
        tuple: Arrays of mu (means), sigma (standard deviations), and pi (cluster probabilities) for each tissue type.
    """
    # Ensure tissue_model has 255 rows corresponding to intensities 1-255
    if len(tissue_model) != 255:
        raise ValueError("Tissue model must have exactly 255 rows, corresponding to intensities 1-255.")
    
    # Convert tissue model to NumPy array for easy computation
    tissue_probabilities = tissue_model.values  # Shape: (255, num_tissues)
    
    # Find the label with the highest probability for each intensity
    label_map = np.argmax(tissue_probabilities, axis=1) + 1  # Labels start from 1

    # Map intensities to labels
    intensities = image.flatten().astype(np.uint8)  # Flattened for easy processing
    labels = np.zeros_like(intensities, dtype=np.uint8)

    # Assign hard labels based on the intensity
    labels[intensities > 0] = label_map[intensities[intensities > 0] - 1]

    # Compute mu, sigma, and pi for each label
    mu = []
    sigma = []
    pi = []
    total_voxels = np.sum(labels > 0)  # Total number of non-background voxels

    for label in range(1, tissue_probabilities.shape[1] + 1):
        # Get all intensities assigned to the current label
        label_intensities = intensities[labels == label]
        
        if len(label_intensities) > 0:
            # Compute mean and standard deviation
            mu.append(np.mean(label_intensities))
            sigma.append(np.std(label_intensities))
            # Compute pi (proportion of voxels in this cluster)
            pi.append(len(label_intensities) / total_voxels)
        else:
            # Handle empty clusters
            mu.append(0)
            sigma.append(0)
            pi.append(0)

    return np.array(mu), np.array(sigma), np.array(pi)


def initialize_clusters(data, n_clusters=3, random_state=0, init_type="kmeans", tissue_model=None, atlas=None):
    """
    Initialize clusters using different strategies.
    """
    print(f"Initializing clusters with init_type={init_type}, n_clusters={n_clusters}")
    
    if init_type == "kmeans":
        # Ensure n_clusters is valid
        if not isinstance(n_clusters, int):
            raise ValueError(f"n_clusters must be an integer. Got {n_clusters}.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
        mu = kmeans.cluster_centers_
        labels = kmeans.labels_
        sigma = np.array([np.std(data[labels == i], axis=0) for i in range(n_clusters)])
        pi = np.ones(n_clusters) / n_clusters

    elif init_type == "tissue_model":
        if tissue_model is None:
            raise ValueError("Tissue model is required for 'tissue_model' initialization.")
        normalized_image = normalize_image(data)
        mu, sigma, pi = compute_mu_sigma_pi_hard_labels(normalized_image, tissue_model)

    elif init_type == "atlas":
        if atlas is None:
            raise ValueError("Atlas is required for 'atlas' initialization.")
        mu = np.mean(atlas, axis=0)[:n_clusters]
        sigma = np.std(atlas, axis=0)[:n_clusters]
        pi = np.ones(n_clusters) / n_clusters

    elif init_type == "atlas_tissue_model":
        if atlas is None or tissue_model is None:
            raise ValueError("Both atlas and tissue model are required for 'atlas_tissue_model' initialization.")
        normalized_image = normalize_image(data)
        mu_tissue, sigma_tissue, pi_tissue = compute_mu_sigma_pi_hard_labels(normalized_image, tissue_model)
        mu_atlas = np.mean(atlas, axis=0)[:n_clusters]
        sigma_atlas = np.std(atlas, axis=0)[:n_clusters]
        pi_atlas = np.ones(n_clusters) / n_clusters
        mu = (mu_tissue + mu_atlas) / 2
        sigma = (sigma_tissue + sigma_atlas) / 2
        pi = (pi_tissue + pi_atlas) / 2

    else:
        raise ValueError(f"Invalid init_type: {init_type}")

    return mu, sigma, pi


# def gaussian_pdf(data, mean, cov):
#     """Calculate Gaussian probability density function with numerical stability."""
#     size = len(data)
#     cov += np.eye(size) * 1e-6  # Add small value to diagonal for stability
#     det = np.linalg.det(cov)
#     norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
#     data_diff = data - mean
#     result = np.exp(-0.5 * np.sum(np.dot(data_diff, np.linalg.inv(cov)) * data_diff, axis=1))
#     return norm_const * result

# Function to calculate the Gaussian probability
def gaussian_pdf(x, mu, cov):
    return multivariate_normal.pdf(x, mean=mu, cov=cov)

def em_algorithm(data, mu, sigma, pi, max_iter=100, tol=1e-6):
    """
    EM algorithm for Gaussian Mixture Models.
    Args:
        data (numpy.ndarray): Input data (n_samples, n_features).
        mu (numpy.ndarray): Initial means for clusters (n_clusters, n_features).
        sigma (numpy.ndarray): Initial standard deviations for clusters (n_clusters, n_features).
        pi (numpy.ndarray): Initial mixing coefficients (n_clusters,).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        tuple: Updated mu, sigma, pi, log_likelihoods, responsibilities.
    """
    n_samples, n_features = data.shape
    n_clusters = mu.shape[0]

    # Validate parameters
    if not isinstance(n_clusters, int):
        raise ValueError(f"n_clusters must be an integer, but got {type(n_clusters)} with value {n_clusters}.")

    print(f"Running EM with n_clusters={n_clusters}, mu={mu}, sigma={sigma}, pi={pi}")

    # Initialize responsibilities (E-step)
    responsibilities = np.zeros((n_samples, n_clusters))
    log_likelihoods = []

    for iter in range(max_iter):
        ### E-Step: Compute responsibilities
        for i in range(n_clusters):
            responsibilities[:, i] = pi[i] * gaussian_pdf(data, mu[i], np.diag(sigma[i]**2))

        # Normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        ### M-Step: Update parameters
        N_k = responsibilities.sum(axis=0)

        # Update mu (means)
        for i in range(n_clusters):
            mu[i] = (responsibilities[:, i].reshape(-1, 1) * data).sum(axis=0) / N_k[i]

        # Update sigma (standard deviations)
        for i in range(n_clusters):
            diff = data - mu[i]
            sigma[i] = np.sqrt((responsibilities[:, i].reshape(-1, 1) * diff**2).sum(axis=0) / N_k[i])

        # Update pi (mixing coefficients)
        pi = N_k / n_samples

        ### Log-Likelihood Calculation
        log_likelihood = np.sum(np.log(np.sum([pi[k] * gaussian_pdf(data, mu[k], np.diag(sigma[k]**2))
                                               for k in range(n_clusters)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iter > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {iter}")
            break

    return mu, sigma, pi, log_likelihoods, responsibilities
