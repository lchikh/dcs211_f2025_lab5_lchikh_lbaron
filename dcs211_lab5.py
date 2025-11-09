import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

###########################################################################


def cleanData(df):
    " Function that takes in a "
    last_column=df.columns[-1]
    print(last_column)
    df_clean=df.drop(columns=[last_column])
    df_array=df.to_numpy()
    return df_clean , df_array

import numpy as np

def predictiveModel(np_training, np_features):
    dist = np.linalg.norm
    num_rows, num_cols = np_training.shape    
    our_features = np_features[0:64]

    closest_features = np_training[0, 0:64]
    closest_digit    = np_training[0, 64]
    closest_distance = dist(our_features - closest_features)
    

    for i in range(1, num_rows):
        current_features = np_training[i, 0:64]
        current_digit    = np_training[i, 64]
        current_distance = dist(our_features - current_features)

        if current_distance < closest_distance:
            closest_distance = current_distance  
            closest_digit    = current_digit

    predicted_digit = closest_digit

    return int(predicted_digit)


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
    features = [0, 0, 9, 16, 16, 16, 5, 0, 0, 1, 14, 10, 8, 16, 8, 0,
 0, 0, 0, 0, 7, 16, 3, 0, 0, 3, 8, 11, 15, 16, 11, 0,
 0, 8, 16, 16, 15, 11, 3, 0, 0, 0, 2, 16, 7, 0, 0, 0,
 0, 0, 8, 16, 1, 0, 0, 0, 0, 0, 13, 10, 0, 0, 0, 0]
#should give back 7
    result = predictiveModel( df_array , np.array(features) )
    print(f"I predict {result} from features {features}")
#defining the features and labels 
    X_all = df_array[:,0:64]  
    y_all = df_array[:,64]  
    print(f"X_all (just features) is \n {X_all}")
    print(f"y_all (just labels)   is \n {y_all}") 
    num_rows = X_all.shape[0]
    print(num_rows)
    test_percent = 0.20
    test_percent = 0.80
    test_size = int(test_percent * num_rows)  
# this first part is when the top 80% is the training set and bottom 20 is testing set
#accuracy = 96.88%
    X_test = X_all[:test_size]    
    y_test = y_all[:test_size]

    X_train = X_all[test_size:]   
    y_train = y_all[test_size:]

# this second part is when the top 20% is the testing set and bottom 20 is training set (just overwriting the values)
#accuracy 91.37%
    X_train = X_all[:num_rows - test_size]
    y_train = y_all[:num_rows - test_size]

    X_test = X_all[num_rows - test_size:]
    y_test = y_all[num_rows - test_size:]

    num_train_rows = len(y_train)
    num_test_rows  = len(y_test)
    print(f"total rows: {num_rows};  training with {num_train_rows} rows;  testing with {num_test_rows} rows" )
    print(f"\t(sanity check:  {num_train_rows} + {num_test_rows} = {num_train_rows + num_test_rows})")
    
    # Loop through each row in the test set
    predictions=[]
    actuals=[]
    np_training = np.hstack((X_train, y_train.reshape(-1, 1)))
    np_testing  = np.hstack((X_test,  y_test.reshape(-1, 1)))
    for i in tqdm(range(len(np_testing)), desc="Predicting digits"): #this is for the progress bar
        test_row = np_testing[i]
        predicted_digit = predictiveModel(np_training, test_row)
        predictions.append(predicted_digit)
        actuals.append(int(test_row[64]))

    accuracy = np.mean(np.array(predictions) == np.array(actuals))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    num_to_draw = 5
    for i in range(num_to_draw):
        #let's grab one row of the df at random, extract/shape the digit to be
         #8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()
    #yes it makes sense, the firt 5 are very unclear what number they are

    # how the data was collected: 
    #link: https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/d1cfd95258db2d252fd921b39805907d_digits_info.txt
    # the sample was was collected from 250 samples by 44 writers
    #WACOM PL-100V pressure sensitive tablet was used with an integrated LCD display and a cordless stylus
    #The tablet sends $x$ and $y$ tablet coordinates and pressure level values of the pen at fixed time intervals (sampling rate) of 100 miliseconds
    #some potential issues: since there is only small amount of writers, the data might not represent the variability of writing styles
    #the small writer-independent test set may not adequately evaluate generalization.
    #The model may not generalize well to other hardware setups
   

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()








