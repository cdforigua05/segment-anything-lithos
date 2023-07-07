import argparse


class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="SAM-Lithos")

        parser.add_argument(
            "--fold", type=int, required=True, help="Cross validation fold"
        )

        parser.add_argument(
            "--multiple_pols",
            action="store_true",
            default=False,
            help="Use multiple polarizations",
        )

        parser.add_argument(
            "--num_pols",
            type=int,
            default=20,
            help="Number of polarizations to use",
        )

        parser.add_argument(
            "--csv",
            type=str,
            default="/media/SSD7/LITHOS/COCO_ANNOTATIONS",
            help="Path to the CSV file",
        )

        parser.add_argument(
            "--image_col",
            type=str,
            default="Sec_patch",
            help="Name of the column on the dataframe that holds the image file names",
        )

        parser.add_argument(
            "--mask_col",
            type=str,
            default="Sec_patch",
            help="the name of the column on the dataframe that holds the mask file names",
        )
        parser.add_argument(
            "--image",
            type=str,
            default="/media/SSD7/LITHOS/SEMANTIC_SEGMENTATION_DATASET",
            help="Path to the input image directory",
        )
        parser.add_argument(
            "--mask",
            type=str,
            default="/media/SSD7/LITHOS/SAM_ANNOTATIONS",
            help="Path to the ground truth mask directory",
        )
        parser.add_argument(
            "--num_epochs", type=int, required=False, default=1, help="number of epochs"
        )
        parser.add_argument(
            "--lr", type=float, required=False, default=3e-4, help="learning rate"
        )
        parser.add_argument(
            "--batch_size", type=int, required=False, default=6, help="batch size"
        )
        parser.add_argument("--model_type", type=str, required=False, default="vit_b")
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="/media/SSD6/pruiz/LITHOS/segment-anything-lithos/segment_anything/models/sam_vit_b_01ec64.pth",
            help="Path to SAM checkpoint",
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            required=True,
            help="Folder to save experiment",
        )
        parser.add_argument(
            "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
        )
        parser.add_argument(
            "--paralelized",
            action="store_true",
            default=False,
            help="Paralelize code in multiple GPUs",
        )
        parser.add_argument(
            "--world_size", type=int, default=3, help="Num of GPUs to use"
        )

        self.args = parser.parse_args()

    def get_args(self):
        return self.args
