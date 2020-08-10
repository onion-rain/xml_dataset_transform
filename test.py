import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2

__all__ = ["draw"]

# def draw(img, detections, save_path):
#     """
#     cv画图
#     args:
#         detections: (x, y, x, y, cls_pred), 单位为像素，
#             注意：x, y为方框中心坐标"""
#     if detections is not None:
#         text_font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         thickness = 2
#         for idx, (x1, y1, x2, y2, cls_pred) in enumerate(detections):
#             x1 = int(x1)
#             y1 = int(y1)
#             x2 = int(x2)
#             y2 = int(y2)
#             # cls_pred = int(cls_pred)
#             cv2.rectangle(img, (x1, y1), (x2, y2), [230, 230, 51], thickness)
#             cv2.putText(img, cls_pred, (x1, y1-4),
#                         text_font, font_scale, bbox_colors[idx], thickness)

#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path , img)

    # return img

def draw(img, detections, save_path):
    """
    args:
        img(3, h, w)
        detections: (x, y, x, y, cls), 单位为像素，
            注意：x, y为方框角坐标"""
    # Create plot
    img = img.transpose(0, -1).transpose(0, 1)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Draw bounding boxes and labels of detection
    if detections is not None:
        # n_cls_preds = len(detections)
        # bbox_colors = random.sample(colors, n_cls_preds)
        for idx, (x1, y1, x2, y2, cls) in enumerate(detections):
            box_w = x2 - x1
            box_h = y2 - y1
            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # color = bbox_colors[idx]
            color = [0.9, 0.9, 0.2, 1]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1-20,
                fontsize=10,
                s=cls,
                color="black",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
