
#%%
import matplotlib.colors as clr
import matplotlib.pyplot as plt


cmRed = clr.LinearSegmentedColormap('red',[(0,0,0),(1,0,0)],256)







#def encoder(img):



#def decoder(img):



def main():


    img = plt.imread(r'C:\Users\guiso\Desktop\MULT\TP1\logo.bmp')

    print(img.shape)


if __name__ == "__main__":

    main()




# %%
