from Cnn import cnnModel

if __name__ == '__main__':

    #  Load images and labels
    pathX = "D:\Head-CT-hemorrhage-detection-master\Dataset\head_ct/*.png"
    pathY = "D:\Head-CT-hemorrhage-detection-master\Dataset\labels.csv"

    cnnModel(140, 140, pathX, pathY)

    
