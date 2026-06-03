import argparse

from synapse.cristae.evaluate import run_cristae_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True,
                        help="Path to GT labels file or directory")
    parser.add_argument("-le", "--labels_ext", default=None,
                        help="Extension of label files when labels_path is a directory")
    parser.add_argument("-k", "--key", default=None,
                        help="H5 dataset key for labels (omit for .tif)")
    parser.add_argument("-s", "--segmentations_path", required=True,
                        help="Path to segmentation file or directory")
    parser.add_argument("-se", "--segmentations_ext", default=None,
                        help="Extension of segmentation files when segmentations_path is a directory")
    parser.add_argument("-sk", "--segmentations_key", default=None,
                        help="H5 dataset key for segmentation (omit for .tif)")
    parser.add_argument("-d", "--dataset_name", default=None,
                        help="Optional name tag for single-file CSV row")
    parser.add_argument("-o", "--output_path", default=None,
                        help="CSV output file or directory")
    parser.add_argument("-hd", "--compute_hd95", default=False, action="store_true",
                        help="Compute HD95 (slow for large volumes)")
    args = parser.parse_args()

    run_cristae_evaluation(
        labels_path=args.labels_path,
        segmentations_path=args.segmentations_path,
        label_key=args.key,
        seg_key=args.segmentations_key,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        labels_ext=args.labels_ext,
        seg_ext=args.segmentations_ext,
        compute_hd95=args.compute_hd95,
    )


if __name__ == "__main__":
    main()
