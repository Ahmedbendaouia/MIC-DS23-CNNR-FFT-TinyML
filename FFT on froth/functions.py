import pandas as pd
import os
import cv2
import random
from pandas import read_excel
from pandas import DataFrame
from sklearn.decomposition import PCA
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def crop(Image, offsetHauteur, offsetLargeur):
    hauteur = Image.shape[0]
    largeur = Image.shape[1]
    croped_image = Image[(hauteur//2 -offsetHauteur):(hauteur//2 +offsetHauteur),(largeur//2 -offsetLargeur):(largeur//2 +offsetLargeur)]
    return croped_image

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    print(pathIn)
    vid= pathIn.split("/")[2].split(".")[0]
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))
        success,image = vidcap.read()
        if(success != False):            
            image = crop(image,200,200)
            cv2.imwrite(pathOut + "/" + vid+"-%d.jpg" % count, image)
            print("frame "+vid+"-%d.jpg" % count,success)
        count = count + 1

def prepareFileAugm(r, video_type):
    writer = pd.ExcelWriter('./data_' + video_type + '_augm_Pb.xlsx', engine = 'xlsxwriter')
    # load the test dataset pb
    test_path = "../" + video_type + "_resizedimages"
    test_names = os.listdir(test_path)

    file_name = 'DataSource.xlsx' # name of your excel file
    
    df = read_excel(file_name, sheet_name = 'Relavage 4')
    name_video=df["points"]
    Analyses = {"points": [],
        "Cu (%)": [],
        "Fe (%)": [],
        "Pb (%)": [],
        "Zn (%)": []
    }

    for video in name_video:
        df_row= df[df['points']== video ]
        if not "-" in video:
            break
        video_0, video_1 = video.split(".")[0].split("-")
        for test_name in test_names:
            if(test_name != "desktop.ini"):
                train_0, train_1, _ = test_name.split(".")[0].split("-")
                if(video_0 == train_0)and (video_1 == train_1):
                    Analyses["points"].append(test_name)
                    Analyses["Cu (%)"].append(df_row["Cu (%)"].to_string(index=False))
                    Analyses["Fe (%)"].append(df_row["Fe (%)"].to_string(index=False))
                    Analyses["Pb (%)"].append(df_row["Pb (%)"].to_string(index=False))
                    Analyses["Zn (%)"].append(df_row["Zn (%)"].to_string(index=False))
    df_Pb = DataFrame(Analyses, columns= ['points', 'Cu (%)', 'Fe (%)', 'Pb (%)', 'Zn (%)']) 
    
    for i in range(0,df_Pb.shape[0]):
        cu = float(df_Pb.iloc[i]['Cu (%)'])
        fe = float(df_Pb.iloc[i]['Fe (%)'])
        pb = float(df_Pb.iloc[i]['Pb (%)'])
        zn = float(df_Pb.iloc[i]['Zn (%)'])
        
        ofsset = random.uniform(0, r)
        df_Pb.at[i,'Cu (%)'] = cu + ofsset
        ofsset = random.uniform(0, r)
        df_Pb.at[i,'Fe (%)'] = fe + ofsset
        ofsset = random.uniform(0, r)
        df_Pb.at[i,'Pb (%)'] = pb + ofsset
        ofsset = random.uniform(0, r)
        df_Pb.at[i,'Zn (%)'] = zn + ofsset
    
    df_Pb.to_excel(writer, sheet_name = 'Relavage 4', index = None, header=True)
    
    writer.save()
    writer.close()

def image_to_fft_pca(image):
    # read the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply 2d fft
    f = np.fft.fft2(img)
    # shift low frequency components to the center of the spectrum
    fshift = np.fft.fftshift(f)
    # fft results are complex numbers
    # so we need to compute the magnitude spectrum of the complex numbers (module in french)
    # mod(z) = |a + bi| = sqrt(a² + b²)
    magnitude_spectrum = np.log(np.sqrt(fshift.real**2 + fshift.imag**2)) # we apply log to make the spectrum more visible
    # apply pca with one component to reduce the dimensionality into one vector
    features = PCA(n_components=1).fit_transform(magnitude_spectrum)
    features = features.reshape(-1)
    return features

def TenneurReel(Video, FileName):
    df = read_excel(FileName, sheet_name = 'Relavage 4')
    CuReel, FeReel, PbReel, ZnReel = 0, 0, 0, 0
    for i in range(0, df.shape[0]):
        if Video == df['points'].values[i]:
            CuReel = df['Cu (%)'].values[i]
            FeReel = df['Fe (%)'].values[i]
            PbReel = df['Pb (%)'].values[i]
            ZnReel = df['Zn (%)'].values[i]
    return CuReel, FeReel, PbReel, ZnReel

def predict_video(video_path, model):
    Teneur_predit = []
    count = 0
    Video = cv2.VideoCapture(video_path)

    while(Video.isOpened()):
        Video.set(cv2.CAP_PROP_POS_MSEC,(count*500))
        success, frame = Video.read()
        if success == False:
            break
        image = crop(frame, 200, 200)
        
        # apply fft
        fft_pca = image_to_fft_pca(image)
        fft_pca = fft_pca.reshape(1, -1)

        #predict
        t = model.predict(fft_pca)

        Teneur_predit.append(t)
        count=count+1

    return Teneur_predit

def cal_quartil(data):
    data = DataFrame(data)
    #find absolute value of z-score for each observation
    z = np.abs(stats.zscore(data))
    #only keep rows in dataframe with all z-scores less than absolute value of 3 
    data_clean = data[(z<3).all(axis=1)]
    #find how many rows are left in the dataframe 
   
    #find Q1, Q3, and interquartile range for each column
    Q1 = data.quantile(q=.25)
    Q3 = data.quantile(q=.75)
    IQR = data.apply(stats.iqr)
    #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
    df = data[~((data < (Q1-1.5*IQR)) | (data > (Q3+1.5*IQR))).any(axis=1)]
    #find how many rows are left in the dataframe
    data_clean = np.average(df)
    
    return data_clean

def CalculMoyenneTenneursPredites(TabTenneursPredites):
    Fe = []
    Cu = []
    Pb = []
    Zn = []
    i = 0
    for i in range(0,len(TabTenneursPredites)):
        Cu.append(TabTenneursPredites[i][0][0])
        Fe.append(TabTenneursPredites[i][0][1])
        Pb.append(TabTenneursPredites[i][0][2])
        Zn.append(TabTenneursPredites[i][0][3])
        
        
        Moyenne_teneur_predit_Pb = cal_quartil(Pb)
        Moyenne_teneur_predit_Zn = cal_quartil(Zn)
        Moyenne_teneur_predit_Cu = cal_quartil(Cu)
        Moyenne_teneur_predit_Fe = cal_quartil(Fe)
        
    return Moyenne_teneur_predit_Cu, Moyenne_teneur_predit_Fe, Moyenne_teneur_predit_Pb, Moyenne_teneur_predit_Zn

def PlotAllTenneurs(TeneurPredit, TeneurReel, Moyenne):
    elements = ['Cu', 'Fe', 'Pb', 'Zn']
    Ordre = {'Cu': 0 , 'Fe': 1, 'Pb': 2 , 'Zn':3}
    
    TeneurPreditList = []
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Real Labels Vs Predicted Labels')

    for ax, Element in zip(axs.flat, elements):
        ax.set_title(Element)
        ax.set(xlabel='Image ID', ylabel='% Value')

        TeneurPreditList = []
        for i in range (0, len(TeneurPredit)):
            TeneurPreditList.append(TeneurPredit[i][0][Ordre[Element]])
                    
        y=TeneurPreditList
        x=TeneurReel[Ordre[Element]]
        Avg_normale = np.average(TeneurPreditList)
        QuartMoyenne = Moyenne[Ordre[Element]]
            
        ax.plot(y, 'ro-')

        ax.axhline(y = x, color = 'b', linestyle = '--')
        ax.axhline(y = QuartMoyenne, color = 'y', linestyle = '--')
        ax.axhline(y = Avg_normale, color = 'g', linestyle = '--')

    fig.legend(['Predicted','Real', 'Moyenne Quartile', 'Moyenne normale'], loc='upper right')
    plt.show()
    
    return