from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

import parse_data as parse
df=parse.df
df = df.dropna(how='any')
df = df.copy()
df['label'] = np.where(df['PM_US_Post'] >35, 1, 0) #0がlow、1がhigh
df_low = df[df['label']==0]
df_high = df[df['label']==1]

x = df[['Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation']]
labels = df['label']


def All():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_jobs=-1)
    
    x = df[['Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation']]
    labels = df['label']
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33, random_state=1)
    
    
    
    mlp    = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=1000)
    mlp.fit(x_train, labels_train)
    
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
  

    scaler = StandardScaler()
    
    np.random.seed(0)
    
    warnings.filterwarnings('ignore')   
    
    def evaluate_alpha(alpha, x, labels, niter):
        scaler = StandardScaler()
        local_crs = []
        for _ in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)

            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            
            mlp = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(50,20), max_iter=1000)
            mlp.fit(x_train_scaled, labels_train)
            
            score = accuracy_score(labels_test, mlp.predict(x_test_scaled))
            local_crs.append(score)
        return local_crs
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 5   
    
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_alpha)(a, x, labels, niter) for a in ALPHA
    )
    
    CR = np.array(results)
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0, h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

def pca_table():
    np.set_printoptions(precision=4, suppress=True)
    x = df[['Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation']]
    labels = df['label']
    pca = PCA(n_components=5)
    pca.fit(x)
    
    print(pca.components_)
    print()
    print(pca.explained_variance_ratio_)

    

np.random.seed(0)
x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

np.set_printoptions(precision=4, suppress=True)
pca_full = PCA(n_components=5)
pca_full.fit(x_train_scaled)

ratios = pca_full.explained_variance_ratio_
components = pca_full.components_

print(ratios)
print(components)

pca_2 = PCA(n_components=2)
x_train_pca = pca_2.fit_transform(x_train_scaled)
x_test_pca = pca_2.transform(x_test_scaled)


ratios_2 = pca_2.explained_variance_ratio_
components_2 = pca_2.components_


        
        
def pca_validation():
    x = df[['Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation']]
    labels = df['label']
    scaler = StandardScaler()

    np.random.seed(0)
    
    warnings.filterwarnings('ignore')   
    
    def evaluate_alpha_pca_adam(alpha, x, labels, niter):
        scaler = StandardScaler()
        local_crs = []
        for _ in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=0.33)
            
       
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
    
            pca = PCA(n_components = 2)
            x_train_pca = pca.fit_transform(x_train_scaled)
            x_test_pca = pca.transform(x_test_scaled)
            
            mlp = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(50,20), max_iter=1000)
            mlp.fit(x_train_pca, labels_train)
            
            score = accuracy_score(labels_test, mlp.predict(x_test_pca))
            local_crs.append(score)
        return local_crs
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
    niter   = 10   
    
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_alpha_pca_adam)(a, x, labels, niter) for a in ALPHA
    )
    
    CR = np.array(results)
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0, h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('alpha', size=16)
    ax.set_ylabel('CR', size=16)
    plt.show()

   

def pca1_pca2(x_train_in, labels_in):
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
            ax.scatter( xx[:,0], xx[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('pca 1')
        ax.set_ylabel('pca 2')
        ax.axis('equal')
        ax.legend()
        
    
    mlp    = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(50, 20), random_state=0, max_iter=1000)
    mlp.fit(x_train_in, labels_in)
    
    
    plt.figure(figsize=(4,8))
    plot_decision_surface(mlp, x_train_in, labels_in, colors=['b','r'], marker_size=20, marker_alpha=1.0)
    #plt.plot(x_test_pca[:,0], x_test_pca[:,1], 'wo',  markersize=5, label='Test set', alpha=0.3)
    plt.legend()
    plt.show()

    


