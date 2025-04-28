# These are some functions to create plots modularly.
# @author Jake Bohman
# @date 7/23/2024
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import easyData as ed

# Creates a frequency histogram
# data: DataFrame containing your data
# category: col of a categorical variable
# x_axis: the col to use for the x axis of the plot
# category_list: list of categories to use
# category_labels: dict of labels for categories
# bins: the number of bins, i.e. how granular the chart is. Sqrt of len(data) is good practice
# title: title
# xlabel: x axis label
# ylabel: y axis label
# ylim: y axis limit
# xlim: x axis limit
# xlower: x axis lower limit
def create_freqhist(data, category, x_axis, bins=10, ylim=1, xlim=50, category_list=None, title=None, category_labels = ed.as2name, xlabel=None, ylabel=None, xlower = 0, ylog=False, xlog=False, fsize=[10,6]):
  plt.figure(figsize=(fsize[0], fsize[1]))
  labels = []
  if(category_list==None):
    category_list = data[category].unique()
  for i in category_list:
    tempData = data[data[category]==i]
    length = len(tempData)
    if(category_labels==None):
        plt.hist(tempData[x_axis], weights=np.ones(length) / length, bins=bins, range=(xlower,xlim),histtype = 'step',label=i)
        labels.append(i)
    else:
        plt.hist(tempData[x_axis], weights=np.ones(length) / length, bins=bins, range=(xlower,xlim),histtype = 'step',label=category_labels[i])
        labels.append(category_labels[i])
  if(ylog):
    plt.yscale('log')
  if(xlog):
     plt.xscale("log")
  plt.title(title if title else f'Histogram of {x_axis}')
  plt.xlabel(xlabel if xlabel else x_axis)
  plt.ylabel(ylabel if ylabel else 'Frequency')
  plt.ylim(0,ylim)
  handles = []
  cmap = plt.cm.tab10
  for i in range (len(category_list)):
     handles.append(mpatches.Patch(color=cmap(i)))
  plt.legend(handles, labels, loc='upper right')
  plt.grid(False)
  plt.show()

# Frequency histogram but with multiple dataframes. Same as above but with lists of data, category, x_axis, category_list
def create_compare_freqhist(datas, x_axes, categories, category_list, bins=10, ylim=1, xlim=50, title=None, category_labels = None, xlabel=None, ylabel=None, xlower = 0, ylog=False, xlog=False, fsize=[10,6]):
  plt.figure(figsize=(fsize[0], fsize[1]))
  for i in range(len(datas)):
     plt.hist(datas[i][datas[i][categories[i]]==category_list[i]][x_axes[i]], weights=np.ones(len(datas[i][datas[i][categories[i]]==category_list[i]])) / len(datas[i][datas[i][categories[i]]==category_list[i]]), bins=bins, range=(xlower,xlim),histtype = 'step',label=category_labels[i])
  if(ylog):
    plt.yscale('log')
  if(xlog):
     plt.xscale("log")
  plt.title(title if title else f'Histogram of {x_axes}')
  plt.xlabel(xlabel if xlabel else x_axes[0])
  plt.ylabel(ylabel if ylabel else 'Frequency')
  plt.ylim(0,ylim)
  plt.grid(False)
  plt.legend(loc='upper right')
  plt.show()

# Defines function to create a boxplot
def create_boxplot(data, y_axis, category, category_labels, ylim=275, category_list=None, title=None, xlabel=None, ylabel=None, tilt=0, ylower=0, showmeans=True, showfliers=False, fsize=[6,6], line=False):
  data_to_plot = []
  ax = plt.figure(figsize=(fsize[0], fsize[1])).add_subplot(111)
  xlabels = []
  if(category_list==None):
     category_list = data[category].unique()
  for i in category_list:
    data_to_plot.append(data[data[category]==i][y_axis])
    if(category_labels==None):
        xlabels.append(i)
    else:
       xlabels.append(category_labels[i])
  ax.set_xticklabels(xlabels)
  bp = ax.boxplot(data_to_plot, patch_artist = True, notch ='True',showfliers=showfliers,showmeans=showmeans)
  plt.title(title if title else y_axis)
  plt.ylabel(ylabel if ylabel else y_axis)
  rotation = tilt if tilt else 0
  plt.xticks(rotation=rotation)
  if(line):
    plt.axhline(0, linestyle='--')
  plt.ylim(ylower, ylim)
  plt.show()

# Defines function to create a boxplot
def create_compare_boxplot(datas, y_axes, categories, category_list, category_labels=None, ylim=275, title=None, xlabel=None, ylabel=None, tilt=0, ylower=0, showmeans=True, showoutliers=False, fsize=[6,6], color=False, colorstep=2, legend=None, line=False):
  data_to_plot = []
  ax = plt.figure(figsize=(fsize[0], fsize[1])).add_subplot(111)
  xlabels = []
  for i in range(len(category_list)):
    data_to_plot.append(datas[i][datas[i][categories[i]]==category_list[i]][y_axes[i]])
    if(category_labels==None):
        xlabels.append(category_list[i])
    else:
        xlabels.append(category_labels[i])
  ax.set_xticklabels(xlabels)
  bp = ax.boxplot(data_to_plot, patch_artist = True, notch ='True',showfliers=showoutliers,showmeans=showmeans)
  plt.title(title if title else y_axes)
  plt.ylabel(ylabel if ylabel else y_axes)
  rotation = tilt if tilt else 0
  plt.xticks(rotation=rotation)
  plt.ylim(ylower, ylim)
  if(color):
     colors = ['pink','lightgreen', 'lightsalmon', 'mediumpurple', 'lightcoral', 'aquamarine']
     count = 0
     for patch in bp['boxes']:
      if(count >= colorstep):
        patch.set(facecolor=colors[int(count/colorstep)-1])
      count += 1
     leg = []
     for j in range(count-1):
        if(j%colorstep==0):
          leg.append(bp['boxes'][j])
     plt.legend(leg, legend)
  if(line):
    plt.axhline(0, linestyle='--')
  plt.show()

# Defines function to create a boxplot over the same data
def create_compare_boxplotSD(data, y_axes, categories, category_list, category_labels=None, ylim=275, title=None, xlabel=None, ylabel=None, tilt=0, ylower=0, showmeans=True, showoutliers=False, fsize=[6,6], color=False, colorstep=2, legend=None, line=False):
  data_to_plot = []
  ax = plt.figure(figsize=(fsize[0], fsize[1])).add_subplot(111)
  xlabels = []
  for i in range(len(category_list)):
    data_to_plot.append(data[data[categories[i]]==category_list[i]][y_axes[i]])
    if(category_labels==None):
        xlabels.append(category_list[i])
    else:
        xlabels.append(category_labels[i])
  ax.set_xticklabels(xlabels)
  bp = ax.boxplot(data_to_plot, patch_artist = True, notch ='True',showfliers=showoutliers,showmeans=showmeans)
  plt.title(title if title else y_axes)
  plt.ylabel(ylabel if ylabel else y_axes)
  rotation = tilt if tilt else 0
  plt.xticks(rotation=rotation)
  plt.ylim(ylower, ylim)
  if(color):
     colors = ['pink','lightgreen', 'lightsalmon', 'mediumpurple', 'lightcoral', 'aquamarine']
     count = 0
     for patch in bp['boxes']:
      if(count >= colorstep):
        patch.set(facecolor=colors[int(count/colorstep)-1])
      count += 1
     leg = []
     for j in range(count-1):
        if(j%colorstep==0):
          leg.append(bp['boxes'][j])
     plt.legend(leg, legend)
  if(line):
    plt.axhline(0, linestyle='--')
  plt.show()

  # Defines function to create a boxplot
def create_time_boxplot(data, ASN, col, title=None, xlabel=None, ylabel=None, order=None, tilt=0, ylim=275):
  data_to_plot = []
  ax = plt.figure(figsize=(10, 6)).add_subplot(111)
  data = data[data['ISP']==ASN]
  ASNlabels = ['Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
  #novdata = data[data['dtime'].str.contains("2022-11")]
  decdata = data[data['dtime'].str.contains("2022-12")]
  jandata = data[data['dtime'].str.contains("2023-01")]
  febdata = data[data['dtime'].str.contains("2023-02")]
  mardata = data[data['dtime'].str.contains("2023-03")]
  aprdata = data[data['dtime'].str.contains("2023-04")]
  maydata = data[data['dtime'].str.contains("2023-05")]
  jundata = data[data['dtime'].str.contains("2023-06")]
  juldata = data[data['dtime'].str.contains("2023-07")]
  #data_to_plot.append(novdata[col])
  data_to_plot.append(decdata[col])
  data_to_plot.append(jandata[col])
  data_to_plot.append(febdata[col])
  data_to_plot.append(mardata[col])
  data_to_plot.append(aprdata[col])
  data_to_plot.append(maydata[col])
  data_to_plot.append(jundata[col])
  data_to_plot.append(juldata[col])
  ax.set_xticklabels(ASNlabels)
  bp = ax.boxplot(data_to_plot, patch_artist = True, notch ='True',showfliers=False,showmeans=True)
  plt.title(title if title else col)
  plt.ylabel(ylabel if ylabel else col)
  rotation = tilt if tilt else 0
  plt.xticks(rotation=rotation)
  plt.ylim(0, ylim)
  print(decdata[col].mean())
  plt.show()

# Defines function to create a boxplot
def create_TOD_boxplot(data, ASN, col, title=None, xlabel=None, ylabel=None, order=None, tilt=0, ylim=275):
  data_to_plot = []
  ax = plt.figure(figsize=(10, 6)).add_subplot(111)
  data = data[data['ISP']==ASN]
  ASNlabels = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00',
               '11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
  zerodata = data[data['dtime'].str.contains(" 00:")]
  onedata = data[data['dtime'].str.contains(" 01:")]
  twodata = data[data['dtime'].str.contains(" 02:")]
  threedata = data[data['dtime'].str.contains(" 03:")]
  fourdata = data[data['dtime'].str.contains(" 04:")]
  fivedata = data[data['dtime'].str.contains(" 05:")]
  sixdata = data[data['dtime'].str.contains(" 06:")]
  sevendata = data[data['dtime'].str.contains(" 07:")]
  eightdata = data[data['dtime'].str.contains(" 08:")]
  ninedata = data[data['dtime'].str.contains(" 09:")]
  tendata = data[data['dtime'].str.contains(" 10:")]
  elevendata = data[data['dtime'].str.contains(" 11:")]
  twelvedata = data[data['dtime'].str.contains(" 12:")]
  thirteendata = data[data['dtime'].str.contains(" 12:")]
  fourteendata = data[data['dtime'].str.contains(" 13:")]
  fifteendata = data[data['dtime'].str.contains(" 14:")]
  sixteendata = data[data['dtime'].str.contains(" 15:")]
  seventeendata = data[data['dtime'].str.contains(" 16:")]
  eighteendata = data[data['dtime'].str.contains(" 17:")]
  nineteendata = data[data['dtime'].str.contains(" 18:")]
  twentydata = data[data['dtime'].str.contains(" 19:")]
  twentyonedata = data[data['dtime'].str.contains(" 20:")]
  twentytwodata = data[data['dtime'].str.contains(" 21:")]
  twentythreedata = data[data['dtime'].str.contains(" 22:")]
  
  #data_to_plot.append(novdata[col])
  data_to_plot.append(zerodata[col])
  data_to_plot.append(onedata[col])
  data_to_plot.append(twodata[col])
  data_to_plot.append(threedata[col])
  data_to_plot.append(fourdata[col])
  data_to_plot.append(fivedata[col])
  data_to_plot.append(sixdata[col])
  data_to_plot.append(sevendata[col])
  data_to_plot.append(eightdata[col])
  data_to_plot.append(ninedata[col])
  data_to_plot.append(tendata[col])
  data_to_plot.append(elevendata[col])
  data_to_plot.append(twelvedata[col])
  data_to_plot.append(thirteendata[col])
  data_to_plot.append(fourteendata[col])
  data_to_plot.append(fifteendata[col])
  data_to_plot.append(sixteendata[col])
  data_to_plot.append(seventeendata[col])
  data_to_plot.append(eighteendata[col])
  data_to_plot.append(nineteendata[col])
  data_to_plot.append(twentydata[col])
  data_to_plot.append(twentyonedata[col])
  data_to_plot.append(twentytwodata[col])
  data_to_plot.append(twentythreedata[col])
  ax.set_xticklabels(ASNlabels)
  bp = ax.boxplot(data_to_plot, patch_artist = True, notch ='True',showfliers=False,showmeans=True)
  plt.title(title if title else col)
  plt.ylabel(ylabel if ylabel else col)
  rotation = tilt if tilt else 0
  plt.xticks(rotation=rotation)
  plt.ylim(0, ylim)
  print(decdata[col].mean())
  plt.show()