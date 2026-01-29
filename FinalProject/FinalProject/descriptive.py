

from IPython.display import display, Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )




import parse_data as parse
df=parse.df





#def extra0():

#　How the PM2.5 concentration fluctuated.(before deleting invalid row)
plt.figure()
plt.plot(df['PM_US_Post'])
plt.show()

plt.figure()
plt.plot(df['PM_US_Post'])
plt.axis([10000,10300, 0, 886])
plt.show()

a = df.isnull().sum()
print(a)

# Since I want to know relationships between weather and PM2.5 concentration, I will delete all row where there is at least one NaN
df = df.dropna(how='any')

print(df.isnull().sum())

def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    return x0, x1, x2


def dispersion(x, print_output=True):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    return y0,y1,y2,y3,y4,y5,y6

def display_central_tendency_table(num=1):
        display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
        df_central = df.apply(lambda x: central(x), axis=0)
        round_dict = {'PM_US_Post':1, 'Hour':0, 'Dew_Point_Temperature':0, 'Humidity':0, 'Pressure':0, 'Temperature':0, 'Precipitation':1}
        df_central = df_central.round( round_dict )
        row_labels = 'mean', 'median', 'mode'
        df_central.index = row_labels
        display( df_central )

display_central_tendency_table(num=1)

def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = {'Hour':1, 'PM_US_Post':1, 'Dew_Point_Temperature':1, 'Humidity':1, 'Pressure':1, 'Temperature':1, 'Precipitation':1}
    df_dispersion         = df.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

display_dispersion_table(num=2)



y    = df['PM_US_Post']
HOUR = df['Hour']
DEWP = df['Dew_Point_Temperature']
HUMI = df['Humidity']
PRES = df['Pressure']
TEMP = df['Temperature']
PREC = df['Precipitation']

def extra1():
    fig,axs = plt.subplots( 1, 6, figsize=(10,3), tight_layout=True )
    axs[0].scatter( DEWP, y, alpha=0.5, color='b' )
    axs[1].scatter( HUMI, y, alpha=0.5, color='r' )
    axs[2].scatter( PRES, y, alpha=0.5, color='g' )
    axs[3].scatter( TEMP, y, alpha=0.5, color='y' )
    axs[4].scatter( PREC, y, alpha=0.5, color='m' )
    axs[5].scatter( HOUR, y, alpha=0.5, color='c' )
    xlabels = 'Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation', 'Hour'
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_ylabel('PM_US Post')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()

    fig,axs = plt.subplots( 1, 1, figsize=(10,3), tight_layout=True )
    axs.scatter( HOUR, y, alpha=0.5, color='c' )

    xlabels = 'Dew_Point_Temperature_Low'
    axs.set_xlabel(xlabels) 

    axs.set_xticks([0,12,24])
    axs.set_ylabel('PM_US_Post')

    [axs.plot(q, y.mean(), 'ro')  for q in range(0, 24)]

    plt.show()

    fig,axs = plt.subplots( 1, 1, figsize=(10,3), tight_layout=True )
    axs.scatter( HOUR, y, alpha=0.5, color='c' )

    xlabels = 'Dew_Point_Temperature_Low'
    axs.set_xlabel(xlabels) 

    axs.set_xticks([0,12,24])
    axs.set_ylabel('PM_US_Post')
    
    [axs.plot(q, y.mean(), 'ro')  for q in range(0, 24)]

    plt.show()

    fig,axs = plt.subplots( 1, 6, figsize=(10,3), tight_layout=True )
    axs[0].scatter( DEWP, y, alpha=0.5, color='b' )
    axs[1].scatter( HUMI , y, alpha=0.5, color='r' )
    axs[2].scatter( PRES, y, alpha=0.5, color='g' )
    axs[3].scatter( TEMP, y, alpha=0.5, color='y' )
    axs[4].scatter( PREC, y, alpha=0.5, color='m' )
    axs[5].scatter( HOUR, y, alpha=0.5, color='c' )
    xlabels = 'Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation', 'Hour'
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_xticks([-40, -20, 0, 20])
    axs[1].set_xticks([0, 50, 100])
    axs[2].set_xticks([1000, 1020, 1040])
    axs[3].set_xticks([-20, 0, 20, 40])
    axs[4].set_xticks([0, 15, 30])
    axs[5].set_xticks([0, 12, 24])
    axs[0].set_ylabel('PM_US_Post')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()

def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)

def extra3():    
    fig,axs = plt.subplots( 1, 6, figsize=(15,3), tight_layout=True )
    ivs     = [DEWP, HUMI, PRES, TEMP, PREC, HOUR]
    colors  = 'b', 'r', 'g', 'y', 'm','c'
    for ax,x,c in zip(axs, ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = 'Dew_Point_Temperature', 'Humidity', 'Pressure', 'Temperature', 'Precipitation', 'Hour'
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_xticks([-40, -20, 0, 20])
    axs[1].set_xticks([0, 50, 100])
    axs[2].set_xticks([1000, 1020, 1040])
    axs[3].set_xticks([-20, 0, 20, 40])
    axs[4].set_xticks([0, 15, 30])
    axs[5].set_xticks([0, 12, 24])
    axs[0].set_ylabel('PM_US_Post')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()

    i_low  = DEWP <= -8  
    i_high = DEWP > -8   
    
    fig,axs = plt.subplots( 1, 2, figsize=(5,3), tight_layout=True )
    
    for ax,i in zip(axs, [i_low, i_high]):
        ax.scatter( DEWP[i], y[i], alpha=0.5, color='b' )
    for ax,i in zip(axs, [i_low, i_high]):
        plot_regression_line(ax, DEWP[i], y[i], color='k', ls='-', lw=2)

    [ax.set_xlabel('Dew_Point_Temperature') for ax in axs] 
    axs[0].set_ylabel('PM_US_Post') 

    axs[0].set_title('Low-DEWP (<= -8)')
    axs[1].set_title('High-DEWP (> -8)')

    axs[0].set_xticks([-40, -32, -24, -16, -8])
    axs[1].set_xticks([-8, 0, 8, 16, 24])
    plt.show

    PM_low     = y <= 70
    PM_high    = y > 70 

    fig,axs = plt.subplots( 1, 2, figsize=(8,3), tight_layout=True )
    PM       = [PREC]
    for ax,PM in zip(axs, [PM_low, PM_high]):
        ax.scatter( PREC[PM], y[PM], alpha=0.5, color='g' )
        plot_regression_line(ax, PREC[PM], y[PM], color='k', ls='-', lw=2)
    [ax.set_xlabel('Precipitation')  for ax in axs] 
    axs[0].set_title('Low-concentration')
    axs[0].set_ylabel('PM_US_Post')
    axs[1].set_title('High-concentration')
    plt.show()

def plot_descriptive():
    fig,axs = plt.subplots( 1, 6, figsize=(15,3), tight_layout=True )
    ivs     = [DEWP, HUMI, PRES, TEMP, PREC, HOUR]
    colors  = 'b', 'r', 'g', 'y', 'm','c'
    for ax,x,c in zip(axs, ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    
    xlabels = 'Dew_Point_Temperature(℃)', 'Humidity(%)', 'Pressure(hPa)', 'Temperature(℃)', 'Precipitation(mm)', 'Hour'
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_xticks([-40, -20, 0, 20])
    axs[1].set_xticks([0, 50, 100])
    axs[2].set_xticks([1000, 1020, 1040])
    axs[3].set_xticks([-20, 0, 20, 40])
    axs[4].set_xticks([0, 15, 30])
    axs[5].set_xticks([0, 12, 24])
    axs[0].set_ylabel('PM_US_Post(μg/m³)')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()


    fig,axs = plt.subplots( 1, 5, figsize=(15,3), tight_layout=True )
    i_low  = DEWP <= -8  
    i_high = DEWP > -8   
    
    for ax,i in zip([axs[0],axs[1]], [i_low, i_high]):
        ax.scatter( DEWP[i], y[i], alpha=0.5, color='b' )
        plot_regression_line(ax, DEWP[i], y[i], color='k', ls='-', lw=2)
        r   = corrcoeff(DEWP[i], y[i])
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))


    [ax.set_xlabel('Dew_Point_Temperature(℃)') for ax in axs] 
    axs[0].set_ylabel('PM_US_Post(μg/m³)') 

    axs[0].set_title('Low-DEWP (<= -8)')
    axs[1].set_title('High-DEWP (> -8)')

    axs[0].set_xticks([-40, -32, -24, -16, -8])
    axs[1].set_xticks([-8, 0, 8, 16, 24])
    
    
    PM_low     = y <= 70
    PM_high    = y > 70 
    PM       = [PREC]
    for ax,PM in zip([axs[2],axs[3]], [PM_low, PM_high]):
        ax.scatter( PREC[PM], y[PM], alpha=0.5, color='g' )
        plot_regression_line(ax, PREC[PM], y[PM], color='k', ls='-', lw=2)
        r   = corrcoeff(PREC[PM], y[PM])
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    [ax.set_xlabel('Precipitation(mm)')  for ax in [axs[2],axs[3]]] 
    axs[2].set_title('Low-concentration(< 70)')
    axs[3].set_title('High-concentration(> 70)')
    axs[2].set_ylabel('PM_US_Post(μg/m³)') 
    axs[3].set_ylabel('PM_US_Post(μg/m³)') 

    
    
    axs[4].scatter( HOUR, y, alpha=0.5, color='c' )
    xlabels = 'Hour'
    axs[4].set_xlabel(xlabels) 

    axs[4].set_xticks([0,12,24])
    axs[4].set_ylabel('PM_US_Post(μg/m³)')

    [axs[4].plot(q, y.mean(), 'ro')  for q in range(0, 24)]


    plt.show

plot_descriptive()

