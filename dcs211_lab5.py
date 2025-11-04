import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random

###########################################################################


def cleanData(df):
    " Function that takes in a "
    last_column=df.columns[-1]
    print(last_column)
    df_clean=df.drop(columns=[last_column])
    df_array=df.to_numpy()
    return df_clean , df_array

def predictModel(np_training, np_features):
    dist = np.linalg.norm
    num_rows, num_cols = np_training.shape    
    our_features = np_features[0:63]
    closest_digit   = np_training[2]
    closest_features = np_training[2,0:63]
    closest_distance = dist(our_features - closest_features)
    
    for i in range(2, num_rows, 1):
        current_digit   = np_training[i]
        current_features = np_training[i,0:63]
        current_distance = dist(our_features - current_features)

        if current_distance < closest_distance:
            closest_distance = current_distance  
            closest_digit   = current_digit

    predicted_digit = closest_digit   
    actual_digit = np_training[64]

    return f" our prediction of {actual_digit} is ({predicted_digit})"







def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

###################
def main() -> None:
    #for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")
    df , df_array = cleanData(df)
    features = [0, 0, 10, 7, 13, 9, 0, 0,
                0, 0, 9, 10, 12, 15, 2, 0,
                0, 0, 4, 11, 10, 11, 0, 0,
                0, 0, 1, 16, 10, 1, 0, 0,
                0, 0, 12, 13, 4, 0, 0, 0,
                0, 0, 12, 1, 12, 0, 0, 0,
                0, 1, 10, 2, 14, 0, 0, 0,
                0, 0, 11, 14, 5, 0, 0, 0]
    result = predictModel( df_array , features )
    print(f"I predict {result} from features {features}")


   # num_to_draw = 5
   # for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
      #  random_row = random.randint(0, len(df) - 1)
       # (digit, pixels) = fetchDigit(df, random_row)

       # print(f"The digit is {digit}")
       # print(f"The pixels are\n{pixels}")  
        #drawDigitHeatmap(pixels)
       # plt.show()

    #
    # OK!  Onward to knn for digits! (based on your iris work...)
    #

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()








