import os
import requests
import argparse
import multiprocessing
from tqdm import tqdm
import pandas as pd

PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
PATH_TRAIN_GALLERY = './dataset/train_gallery_file.csv'


def pid_to_url_filename(pid_list, output_path):
    return map(lambda pid: (f"https://m.media-amazon.com/images/I/{pid}.jpg", os.path.join(output_path, f"{pid}.jpg")), pid_list)


def download_image(url_filename):
    url, filename = url_filename

    dir_name = os.path.dirname(filename)
    os.makedirs(dir_name, exist_ok=True)

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(filename, "wb") as fout:
            for chunk in r.iter_content(chunk_size=1024):
                fout.write(chunk)


def download(args):
    if args.annotation:
        annotation = pd.read_csv(PATH_APPAREL_TRAIN_ANNOTATION)
        non_targets = annotation['Non-Target Image ID']
        targets = annotation['Target Image ID']
        sources = annotation['Source Image ID']

        annotate_concat = pd.concat(
            [sources, targets, non_targets], ignore_index=True).drop_duplicates()
        annotation_download_list = annotate_concat.tolist()

        print("Downloading annotation images from",
              PATH_APPAREL_TRAIN_ANNOTATION)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(download_image, pid_to_url_filename(annotation_download_list, "images/annotation")), total=len(annotation_download_list)):
                pass

        print("DONE!\n")

    if args.query:
        query = pd.read_json(PATH_QUERY_FILE, lines=True)
        candidates = query['candidates'].apply(pd.Series).stack().reset_index(
            drop=True).map(lambda x: x['candidate_pid']).drop_duplicates()
        source_pids = query['source_pid']

        query_concat = pd.concat(
            [candidates, source_pids], ignore_index=True).drop_duplicates()
        query_download_list = query_concat.tolist()

        print("Downloading query images from", PATH_QUERY_FILE)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(download_image, pid_to_url_filename(query_download_list, "images/query")), total=len(query_download_list)):
                pass

        print("DONE!\n")

    if args.gallery:
        gallery = pd.read_csv(PATH_TRAIN_GALLERY)
        gallery_images = gallery['Image ID']
        gallery_download_list = gallery_images.tolist()

        print("Downloading gallery images from", PATH_APPAREL_TRAIN_ANNOTATION)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(download_image, pid_to_url_filename(gallery_download_list, "images/gallery")), total=len(gallery_download_list)):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download images from amazon server')
    parser.add_argument('--annotation', action="store_true")
    parser.add_argument('--query', action="store_true")
    parser.add_argument('--gallery', action="store_true")
    args = parser.parse_args()
    download(args)
