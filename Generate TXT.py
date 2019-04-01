import os



GROUNDTRUTH_PATH = "./Dataset/groundTruth"
IMAGE_PATH = "./Dataset/images"
OUTPUT_PATH = "./Dataset"



class MAIN():
    def __init__(self,image_path,groundtruth_path):
        dics = ["test","train","val"]
        self.image_path = image_path
        self.groundtruth_path = groundtruth_path
        for i in dics:
            image_files = os.listdir(self.image_path + '/' + i)
            groundtruth_files = os.listdir(self.groundtruth_path + '/' + i)
            with open(OUTPUT_PATH + '/' + i + '.txt','w') as f:
                for j in range(len(image_files)):
                    f.writelines('./images/' + i + '/' + image_files[j] + " " + './groundfiles/' + i + '/' + groundtruth_files[j])
                    f.write('\n')




if __name__ == '__main__':
    t = MAIN(IMAGE_PATH,GROUNDTRUTH_PATH)
