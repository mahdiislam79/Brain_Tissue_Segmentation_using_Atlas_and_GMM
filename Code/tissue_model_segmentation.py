import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


class TissueModelSegmentation:
    def __init__(self, vec_images, vec_gt, tissue_models_map):
        self.vec_images = vec_images
        self.vec_gt = vec_gt
        self.tissue_models_map = tissue_models_map
        self.metrics_df = None  # To store Dice scores for each image

    # Helper function for normalization
    @staticmethod
    def normalize_image(vec, new_min=0, new_max=255):
        min_val = np.min(vec)
        max_val = np.max(vec)
        normalized_vec = (vec - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        return normalized_vec.astype(np.uint8)

    # Function to apply tissue model to the image, skipping background
    def apply_tissue_model(self, image):
        seg_image = image.copy()
        for i in range(1, 256):  # Start from 1, skipping 0 as background
            seg_value = self.tissue_models_map[i - 1]  # Adjust index since tissue_models_map starts from 1
            seg_image[seg_image == i] = seg_value
        return seg_image

    # Function to calculate Dice scores for each label
    @staticmethod
    def calculate_dice_scores(pred, gt, labels=[1, 2, 3]):
        dice_scores = {}
        for label in labels:
            pred_count = np.sum(pred == label)
            gt_count = np.sum(gt == label)
            intersection = np.sum((pred == label) & (gt == label))
            if pred_count + gt_count == 0:
                dice_score = 1.0  # Perfect match if label is absent in both
            else:
                dice_score = 2 * intersection / (pred_count + gt_count)
            dice_scores[label] = dice_score
        return dice_scores

    # Main segmentation function
    def segment_images(self):
        metrics = {'Image': [], 'CSF': [], 'WM': [], 'GM': []}

        for i, image in enumerate(self.vec_images):
            gt = self.vec_gt[i]
            image = np.where(gt > 0, image, 0)
            masked_image = image[gt > 0]
            image[gt > 0] = self.normalize_image(masked_image)

            # Apply tissue model
            segmented_image = np.zeros(image.shape, dtype=np.int16)
            segmented_image[gt > 0] = self.apply_tissue_model(image[gt > 0])

            # Compute Dice scores
            dice_scores = self.calculate_dice_scores(segmented_image, gt, labels=[1, 2, 3])
            metrics['Image'].append(f'Image_{i+1}')
            metrics['CSF'].append(dice_scores[1])
            metrics['WM'].append(dice_scores[2])
            metrics['GM'].append(dice_scores[3])

        self.metrics_df = pd.DataFrame(metrics)

    # Function to create a table for Dice scores
    def create_dice_score_table(self):
        if self.metrics_df is None:
            self.segment_images()  # Generate metrics if not already done
        return self.metrics_df

    # Function to create a color-coded boxplot of Dice scores without black background
    def plot_dice_score_boxplot(self):
        if self.metrics_df is None:
            self.segment_images()  # Generate metrics if not already done

        # Reshape the data for Seaborn
        melted_df = self.metrics_df.melt(id_vars=['Image'], value_vars=['CSF', 'WM', 'GM'],
                                         var_name='Tissue', value_name='Dice Score')

        # Define softer colors for each tissue type
        tissue_palette = {'CSF': '#a8dadc', 'WM': '#457b9d', 'GM': '#1d3557'}

        # Plot with Seaborn
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=melted_df, x='Tissue', y='Dice Score', palette=tissue_palette)
        plt.title("Dice Score Distribution for CSF, WM, and GM Using Tissue Models")
        plt.ylabel("Dice Score")
        plt.xlabel("Tissue Type")

        # Add legend
        legend_handles = [mpatches.Patch(color=color, label=tissue) for tissue, color in tissue_palette.items()]
        plt.legend(handles=legend_handles, title="Tissue Types")

        plt.show()

    # Function to visualize a 3x3 grid of GT vs. prediction with consistent dimensions and black background
    def plot_comparison_grid(self):
        fig, axes = plt.subplots(3, 6, figsize=(18, 12), facecolor='black')  # 3x6 grid for side-by-side GT and Pred

        slice_indices = [self.vec_images[0].shape[2] // 2, self.vec_images[0].shape[1] // 2, self.vec_images[0].shape[0] // 2]

        # Define softer colors for tissue types
        tissue_colors = {1: '#a8dadc', 2: '#457b9d', 3: '#1d3557'}  # Softer colors for CSF, WM, GM
        cmap = plt.cm.colors.ListedColormap(['black'] + list(tissue_colors.values()))

        for ax in axes.ravel():
            ax.set_facecolor('black')  # Set background color for each subplot
            ax.axis('off')

        for i, image in enumerate(self.vec_images[:3]):  # Limit to the first 3 images
            gt = self.vec_gt[i]
            image = np.where(gt > 0, image, 0)
            masked_image = image[gt > 0]
            image[gt > 0] = self.normalize_image(masked_image)
            segmented_image = np.zeros(image.shape, dtype=np.int16)
            segmented_image[gt > 0] = self.apply_tissue_model(image[gt > 0])

            for j, (view, slice_index) in enumerate(zip(['Axial', 'Coronal', 'Sagittal'], slice_indices)):
                if view == 'Axial':
                    gt_slice = gt[:, :, slice_index]
                    pred_slice = segmented_image[:, :, slice_index]
                elif view == 'Coronal':
                    gt_slice = gt[:, slice_index, :]
                    pred_slice = segmented_image[:, slice_index, :]
                elif view == 'Sagittal':
                    gt_slice = gt[slice_index, :, :]
                    pred_slice = segmented_image[slice_index, :, :]

                # Ground Truth
                axes[i, j * 2].imshow(gt_slice, cmap=cmap, vmin=0, vmax=3)
                axes[i, j * 2].set_title(f"Image {i+1} - {view} GT", color='white')
                axes[i, j * 2].axis('off')

                # Prediction
                axes[i, j * 2 + 1].imshow(pred_slice, cmap=cmap, vmin=0, vmax=3)
                axes[i, j * 2 + 1].set_title(f"Image {i+1} - {view} Pred", color='white')
                axes[i, j * 2 + 1].axis('off')

        # Add a suptitle for the entire grid
        fig.suptitle("Comparison of Ground Truth and Prediction for Different Views", fontsize=16, color='white')

        # Add a color legend with white text for better visibility
        legend_handles = [mpatches.Patch(color=color, label=tissue) for tissue, color in zip(['CSF', 'WM', 'GM'], tissue_colors.values())]
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, title="Tissue Types", fontsize=12, facecolor='black', edgecolor='white', frameon=True, title_fontsize='13', labelcolor='white')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to fit suptitle and legend
        plt.show()