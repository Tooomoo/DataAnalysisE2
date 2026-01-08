
from IPython.display import display, Markdown 
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import parse_data as parse
df=parse.df
df = df.dropna(how='any')

def glaph_PM():
    plt.figure()
    plt.plot(df['PM_US_Post'])
    plt.axis([10000,10300, 0, 886])
    plt.show()

    x  = np.linspace(10000, 10300, 301)    
    X    = np.array([x]).T
    knr  = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')
    y = df['PM_US_Post'][10000:10301]
    knr.fit(X, y)


    # predict the target values:
    xi   = np.linspace(10000, 10300, 500) # feature values for prediction
    Xi   = np.array([xi]).T
    yi   = knr.predict(Xi)              # predicted target values


    # plot:
    plt.figure()
    plt.plot(x, y, 'b.', label='Observations')
    plt.plot(xi, yi, 'r-', label='Predictions')
    plt.xlabel('Feature 1 value')
    plt.ylabel('Target value')
    plt.legend()
    plt.show()

    x  = np.arange(25162)
    X    = np.array([x]).T
    knr  = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
    y = df['PM_US_Post']
    knr.fit(X, y)


    # predict the target values:
    xi   = np.linspace(0, 25162, 50000) # feature values for prediction
    Xi   = np.array([xi]).T
    yi   = knr.predict(Xi)              # predicted target values


    # plot:
    plt.figure()
    plt.plot(x, y, 'b.', label='Observations')
    plt.plot(xi, yi, 'r-', label='Predictions')
    plt.xlabel('Feature 1 value')
    plt.ylabel('Target value')
    plt.legend()
    plt.show()



df = df.copy()
df['label'] = np.where(df['PM_US_Post'] >35, 1, 0) #0がlow、1がhigh
plt.figure(figsize=(8,8))
ax = plt.axes()
df_low = df[df['label']==0]
df_high = df[df['label']==1]

def HUMI_TEMP():
    #x = df[[df.iloc[:,3],df.iloc[:,5]]]#dfをこの機械学習向けに
    x = df[['Humidity', 'Temperature']]
    #print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)

    #print( f'Training data shape:    {x_train.shape}'  )
#print( f'Test data shape:        {x_test.shape}'  )
    #print( f'Training labels shape:  {labels_train.shape}'  )
    #print( f'Test labels shape:      {labels_test.shape}'  )

    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()

        # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    ## plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)

    from sklearn.preprocessing import StandardScaler


    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)

    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(50, 20), max_iter=500, random_state=1)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()
        
    # create and train a classifier:
    #mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    #mlp.fit(x_train, labels_train)


    ## plot the decision surface:
    #plt.figure(figsize=(8,8))
    #plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    #plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    #plt.legend()
    #plt.show()

    #labels_pred_train = mlp.predict(x_train)
    #labels_pred_test  = mlp.predict(x_test)
    #cr_train          = accuracy_score(labels_train, labels_pred_train)
   # cr_test           = accuracy_score(labels_test, labels_pred_test)
   #print( f'Classification rate (training) = {cr_train}' )
    #print( f'Classification rate (test)     = {cr_test}' )
 #plt.figure()
    #ax = plt.axes()
    #h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    #h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    #ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    #ax.set_xlabel('alpha', size=16)
    #ax.set_ylabel('CR', size=16)
    #plt.show()

def HUMI_TEMP_70():

#基準値を７０に
    df = df.copy()
    df['label'] = np.where(df['PM_US_Post'] >70, 1, 0) #0がlow、1がhigh
    plt.figure(figsize=(8,8))
    ax = plt.axes()
    df_low = df[df['label']==0]
    df_high = df[df['label']==1]
    ax.plot( df_low.iloc[:,3], df_low.iloc[:,5], 'bo', ms=10, label='Low_PM' )
    ax.plot( df_high.iloc[:,3], df_high.iloc[:,5], 'ro', ms=10, label='High_PM' )
    ax.legend()
    plt.show()

    #x = df[[df.iloc[:,3],df.iloc[:,5]]]#dfをこの機械学習向けに
    x = df[['Humidity', 'Temperature']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)

    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )

    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
            nlabels   = np.unique( labels ).size
            colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
            ax        = plt.gca() if (ax is None) else ax
            xmin,xmax = x.min(axis=0), x.max(axis=0)
            Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
            xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
            labelsp   = classifier.predict(xp)
            Labelsp   = np.reshape(labelsp, Xp.shape)
            cmap      = ListedColormap(colors)
            for i,label in enumerate( np.unique(labels) ):
                xx   = x[labels==label]
                ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
            plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.axis('equal')
            ax.legend()
    

    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)


    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()

    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )

       

    from sklearn.preprocessing import StandardScaler


    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    mlp.fit(x_train_scaled, labels_train)

    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn

    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value

    np.random.seed(0)

    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()


def HUMI_PRES():
    df = df.copy()
    df['label'] = np.where(df['PM_US_Post'] >35, 1, 0) #0がlow、1がhigh
    plt.figure(figsize=(8,8))
    ax = plt.axes()
    df_low = df[df['label']==0]
    df_high = df[df['label']==1]
    ax.plot( df_low.iloc[:,3], df_low.iloc[:,5], 'bo', ms=10, label='Low_PM' )
    ax.plot( df_high.iloc[:,3], df_high.iloc[:,5], 'ro', ms=10, label='High_PM' )
    ax.legend()
    plt.show()

    #x = df[[df.iloc[:,3],df.iloc[:,5]]]#dfをこの機械学習向けに
    x = df[['Humidity', 'Pressure']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=3, hidden_layer_sizes=(50, 20), random_state=0)
    mlp.fit(x_train, labels_train)
    
    
    # calculate the CRs for the training and test sets":
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def HUMI_PREC():

    x = df[['Humidity', 'Precipitation']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    
    # create and train a classifier:
    #mlp    = MLPClassifier(solver='lbfgs', alpha=3, hidden_layer_sizes=(50, 20), random_state=0)
    #mlp.fit(x_train, labels_train)
    
    #
    # calculate the CRs for the training and test sets":
    #labels_pred_train = mlp.predict(x_train)
    #labels_pred_test  = mlp.predict(x_test)
    #cr_train          = accuracy_score(labels_train, labels_pred_train)
    #cr_test           = accuracy_score(labels_test, labels_pred_test)
    #print( f'Classification rate (training) = {cr_train}' )
    #print( f'Classification rate (test)     = {cr_test}' )
    
    
    # plot the decision surface:
    #plt.figure(figsize=(8,8))
    #plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    #plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    #plt.legend()
    #plt.show()
    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)

    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def HUMI_DEWP():

    x = df[['Humidity', 'Dew_Point_Temperature']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    

    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def PRES_DEWP():
    x = df[['Pressure', 'Dew_Point_Temperature']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    
    
    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def TEMP_DEWP():
    x = df[['Temperature', 'Dew_Point_Temperature']]
    #print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    #print( f'Training data shape:    {x_train.shape}'  )
    #print( f'Test data shape:        {x_test.shape}'  )
    #print( f'Training labels shape:  {labels_train.shape}'  )
    #print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    #print( f'Classification rate (training) = {cr_train}' )
    #print( f'Classification rate (test)     = {cr_test}' )
    
    
    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def TEMP_PRES():

    
    #x = df[[df.iloc[:,3],df.iloc[:,5]]]#dfをこの機械学習向けに
    x = df[['Temperature', 'Pressure']]
    print(x)
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    print( f'Training data shape:    {x_train.shape}'  )
    print( f'Test data shape:        {x_test.shape}'  )
    print( f'Training labels shape:  {labels_train.shape}'  )
    print( f'Test labels shape:      {labels_test.shape}'  )
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            xx   = x[labels==label]
            ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        
    
    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=30000)
    mlp.fit(x_train, labels_train)
    
    
    # plot the decision surface:
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test.iloc[:,0], x_test.iloc[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) = {cr_train}' )
    print( f'Classification rate (test)     = {cr_test}' )
    
    
    
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
    
    mlp.fit(x_train_scaled, labels_train)
    
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            mlp    = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(10,5 ), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
    
    CR      = np.array(CR)
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()


    


        
        
    
    
    
        
        