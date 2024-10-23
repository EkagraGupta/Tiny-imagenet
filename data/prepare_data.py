import os
import urllib.request
import zipfile
from random import shuffle
from math import floor

def download_dataset():
    print('Downloading dataset...')
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    path = "%s/data/tiny-image-200.zip" % os.getcwd()
    urllib.request.urlretrieve(url, path)
    print('Download completed!')

def unzip_dataset():
    path_to_zip_file = "%s/data/tiny-image-200.zip" % os.getcwd()
    directory_to_extract_to = "%s/data" % os.getcwd()
    print("Extracting zip file: %s" % path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print("Exacted at: %s" % directory_to_extract_to)

def format_val():
    val_dir = "%s/data/tiny-imagenet-200/val" % os.getcwd()
    print("Formatting: %s"  % val_dir)
    val_annotations = "%s/val_annotations.txt" % val_dir
    val_dict = {}

    with open(val_annotations, 'r') as f:
        for line in f:
            line = line.strip().split()
            assert(len(line) == 6)
            wnind = line[1]
            im_name = line[0]
            boxes = '\t'.join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((im_name, boxes))
    assert(len(val_dict)==200)
    for wnind, entries in val_dict.items():
        val_wnind_dir = "%s/%s" % (val_dir, wnind)
        val_images_dir = "%s/images" % val_dir
        val_wnind_images_dir = "%s/images" % val_wnind_dir
        # print("val_wnind_dir: %s" % val_wnind_dir)
        # print("val_images_dir: %s" % val_images_dir)
        # print("val_wnind_images_dir: %s" % val_wnind_images_dir)
        # break
        if not os.path.exists(val_wnind_dir):
            os.mkdir(val_wnind_dir)
        if not os.path.exists(val_images_dir):
            os.mkdir(val_images_dir)
        if not os.path.exists(val_wnind_images_dir):
            os.mkdir(val_wnind_images_dir)
        wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
        f = open(wnind_boxes, "w")
        for im_name, box in entries:
            source = "%s/%s" % (val_images_dir, im_name)
            dest = "%s/%s" % (val_wnind_images_dir, im_name)
            os.system("cp %s %s" % (source, dest))
            f.write("%s\t%s\n" % (im_name, box))
        #     break
        # break
        f.close()
    
    os.system("rm -rf %s" % val_images_dir)
    print("Cleaning up: %s" % val_images_dir)
    print("Formatted val data done")


def split_train_test():
    split_quota = 0.7
    print("Splitting Train+Val into %s-%s" % (split_quota*100, (1 - split_quota)*100))
    base_dir = "%s/data/tiny-imagenet-200" % os.getcwd()
    train_dir = "%s/train" % base_dir
    val_dir = "%s/val" % base_dir
    fwnind = "%s/wnids.txt" % base_dir
    wninds = set()
    with open(fwnind, 'r') as f:
        for wnind in f:
            wninds.add(wnind.strip())
    
    assert(len(wninds) == 200)
    new_train_dir = "%s/new_train" % base_dir
    new_test_dir = "%s/new_test" % base_dir
    if not os.path.exists(new_train_dir):
        os.mkdir(new_train_dir)
    if not os.path.exists(new_test_dir):
        os.mkdir(new_test_dir)
    
    total_train, total_test = 0, 0
    for wnind in wninds:
        wnind_train, wnind_test = 0, 0
        new_train_wnind_dir = "%s/%s" % (new_train_dir, wnind)
        new_test_wnind_dir = "%s/%s" % (new_test_dir, wnind)
        if not os.path.exists(new_train_wnind_dir):
            os.mkdir(new_train_wnind_dir)
        if not os.path.exists(new_test_wnind_dir):
            os.mkdir(new_test_wnind_dir)
        if not os.path.exists(new_train_wnind_dir+"/images"):
            os.mkdir(new_train_wnind_dir+"/images")
        if not os.path.exists(new_test_wnind_dir+"/images"):
            os.mkdir(new_test_wnind_dir+"/images")
        new_train_wnind_boxes = "%s/%s_boxes.txt" % (new_train_wnind_dir, wnind)
        f_train = open(new_train_wnind_boxes, "w")
        new_test_wnind_boxes = "%s/%s_boxes.txt" % (new_test_wnind_dir, wnind)
        f_test = open(new_test_wnind_boxes, "w")

        dirs = [train_dir, val_dir]
        for wdir in dirs:
            wnind_dir = "%s/%s" % (wdir, wnind)
            wnind_boxes = "%s/%s_boxes.txt" % (wnind_dir, wnind)
            ims = []
            with open(wnind_boxes, "r") as f:
                for line in f:
                    line = line.strip().split()
                    im_name = line[0]
                    boxes = "\t".join(line[1:])
                    ims.append((im_name, boxes))
            print("[Old] wnd: %s - #: %s" % (wnind, len(ims)))
            shuffle(ims)
            split_n = floor(len(ims)*0.7)
            train_ims = ims[:split_n]
            test_ims = ims[split_n:]

            for im_name, box in train_ims:
                source = "%s/images/%s" % (wnind_dir, im_name)
                dest = "%s/images/%s" % (new_train_wnind_dir, im_name)
                os.system("cp %s %s" % (source, dest))
                f_train.write("%s\t%s\n" % (im_name, box))
                wnind_train += 1
            
            for im_name, box in test_ims:
                source = "%s/images/%s" % (wnind_dir, im_name)
                dest = "%s/images/%s" % (new_test_wnind_dir, im_name)
                os.system("cp %s %s" % (source, dest))
                f_test.write("%s\t%s\n" % (im_name, box))
                wnind_test += 1
            
        f_train.close()
        f_test.close()
        print("[New] wnind: %s - #train: %s - #test: %s" % (wnind, wnind_train, wnind_test))

        total_train += wnind_train
        total_test += wnind_test
    
    print(f"[New] #train: {total_train} - #test: {total_test}")
    os.system(f"rm -rf {train_dir}")
    os.system(f"rm -rf {val_dir}")
    print(f"Cleaning up: {train_dir}")
    print(f"Cleaning up: {val_dir}")
    print(f"Creating new train data at: {new_train_dir}")
    print(f"Creating new test data at: {new_test_dir}")
    print("Splitting done")













if __name__=='__main__':
    download_dataset()
    unzip_dataset()
    format_val()
    split_train_test()