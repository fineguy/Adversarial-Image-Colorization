import argparse
import os
import random
import time
import urllib

from multiprocessing.pool import ThreadPool

import flickrapi


def fetch_url(data):
    idx, url = data
    try:
        urllib.request.urlretrieve(url, '{}.jpg'.format(idx))
    except Exception as e:
        print("Failed on idx {} url: {}".format(idx, url), e)


def random_split(seq, ratio):
    """Randomly split a list into two parts in the given ratio and preserving the ordering.
    This is done in O(N)-time and O(1)-auxilary-space way. Refer to
    https://stackoverflow.com/questions/6482889/get-random-sample-from-list-while-maintaining-ordering-of-items
    """
    length = len(seq)
    left_length = int(length * ratio)
    left, right = [], []
    left_picked = 0
    for i, elem in enumerate(seq):
        prob = (left_length - left_picked) / (length - i)
        if random.random() < prob:
            left.append(elem)
            left_picked += 1
        else:
            right.append(elem)
    return left, right


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrap photos from flickr')
    parser.add_argument('-k', '--key', required=True, help='API key')
    parser.add_argument('-s', '--secret', required=True, help='API secret')
    parser.add_argument('-d', '--dir', default='../../data/raw', help='output directory')
    parser.add_argument('-q', '--query', default='lake baikal', help='keyword query')
    parser.add_argument('-t', '--threads', type=int, default=50, help='# of downloading threads')
    parser.add_argument('--train', type=float, default=0.7, help='training data ratio')
    parser.add_argument('--valid', type=float, default=0.2, help='validation data ratio')
    parser.add_argument('--test', type=float, default=0.1, help='testing data ratio')
    parser.add_argument('--seed', type=int, default=1334, help='random seed')
    args = parser.parse_args()

    assert abs(args.train + args.valid + args.test - 1.0) <= 1e-09, "Make sure all ratios sum up to 1."

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    os.chdir(args.dir)
    random.seed(args.seed)

    print("Searching for photos ...")
    flickr = flickrapi.FlickrAPI(args.key, args.secret)
    photos = flickr.walk(text=args.query, tag_mode='all', tags=args.query,
                         extras='url_m', per_page=100, sort='relevance')
    urls = [photo.get('url_m') for photo in photos]
    print("Found {} photos matching your query.".format(len(urls)))

    train_urls, urls = random_split(urls, args.train)
    valid_urls, test_urls = random_split(urls, args.valid / (args.valid + args.test))

    for dir_name, urls in zip(['train', 'valid', 'test'], [train_urls, valid_urls, test_urls]):
        print("Downloading {} photos for {}.".format(len(urls), dir_name))
        os.mkdir(dir_name)
        os.chdir(dir_name)
        results = ThreadPool(args.threads).imap_unordered(fetch_url, enumerate(urls))
        start_time = time.time()
        for i, _ in enumerate(results):
            if i % 1000 == 0:
                print("Downloaded {} photos in {:.2f} seconds.".format(i, time.time() - start_time))

        print("Finished downloading images in {:.2f}.".format(time.time() - start_time))
        os.chdir("../")
