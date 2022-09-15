from model import Inference
import shutil, os
from skimage.io import imread, imsave
import threading
import time

dataset = r'E:\datasets\Market-1501-v15.09.15'
ori_folder = [
    os.path.join(dataset, 'bounding_box_train'),
    os.path.join(dataset, 'bounding_box_test'), 
    os.path.join(dataset, 'query')
]

cor_folder = [
    os.path.join(dataset, 'bounding_box_train_cor'),
    os.path.join(dataset, 'bounding_box_test_cor'), 
    os.path.join(dataset, 'query_cor')
]

# Clean folders
try:
    for folder in cor_folder:
        shutil.rmtree(folder)
        print("Remove ", folder)
except Exception as e:
    print(e)

for folder in cor_folder:
    os.mkdir(folder)
    print("Create ", folder)

total_img = 0

class CorrectThread (threading.Thread):
    def __init__(self, base_in, base_out, part, n_part):
        threading.Thread.__init__(self)
        self.base_in = base_in
        self.base_out = base_out
        self.imgs = [file for file in os.listdir(base_in) if file.endswith(".jpg")]
        self.part = part
        self.n_part = n_part

    def run(self):
        n = len(self.imgs)
        for i, name in enumerate(self.imgs):
            if i % self.n_part == self.part:
                # Get img file path
                imgpath = os.path.join(self.base_in, name)
                print(f"[{i+1}/{n}] {imgpath}")

                # Process
                I_in = imread(imgpath) / 255
                I_out = Inference.infer(I_in) * 255
                I_out = I_out.astype('uint8')
                imsave(os.path.join(self.base_out, name), I_out)

                global total_img
                total_img += 1

threads = []

t1 = time.time()

for folder_index in range(len(ori_folder)):
    # Create new threads
    n_part = 5
    for i in range(n_part):
        thread = CorrectThread(ori_folder[folder_index], cor_folder[folder_index], i, n_part)
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for t in threads:
        t.join()

t2 = time.time()

timelapse = t2 - t1
print("Running time: ", timelapse)
print(f"Avg: {timelapse / total_img} s / img")