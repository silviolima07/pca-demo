import streamlit as st

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

from PIL import Image

pd.set_option('precision',2)


from bokeh.models.widgets import Div

# Cache
@st.cache
def read_digits():
    return load_digits()

def plot_digits(digits):
    #fig = plt.figure( figsize=(15,15))
    #fig.subplots_adjust(hspace=0.95, wspace=0.05)
    #for i in range(40):
    #    ax =fig.add_subplot(10,10,1+i, xticks=[], yticks=[])
    #    ax.imshow(digits.images[i],cmap=plt.cm.gray_r, interpolation='nearest')
    #    ax.text(0,9, str(digits.target[i]))
    #st.pyplot()
    fig, axes = plt.subplots(4,10, figsize=(10, 4), subplot_kw= { 'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace = 0.95, wspace=0.05))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i].reshape(8,8),cmap='binary', interpolation='nearest', clim=(0,16))
        ax.text(0,9, str(digits.target[i]))
    st.pyplot()

def plot_noisy(noisy,digits):
    fig, axes = plt.subplots(4,10, figsize=(10, 4), subplot_kw= { 'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace = 0.95, wspace=0.05))
    for i, ax in enumerate(axes.flat):
        ax.imshow(noisy[i].reshape(8,8),cmap='binary', interpolation='nearest', clim=(0,16))
        ax.text(0,9, str(digits.target[i]))
    st.pyplot()


def plot_pca(project,digits):
    fig = plt.figure( figsize=(10,10))
    x = project[:,0]
    y = project[:,1]
    plt.title("Two Components / Dimensions")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    fig = plt.scatter(x = x, y = y, c=digits.target, cmap="Paired")
    plt.colorbar()
    st.pyplot()

def main():
    """Principal Components Analysis  App """

    #logo = Image.open("logo.png")
    #st.sidebar.image(logo,caption="", width=200)
   

    html_page = """
    <div style="background-color:tomato;padding=50px">
        <p style='text-align:center;font-size:50px;font-weight:bold'>PCA - Principal Component Analysis</p>
    </div>
              """
    st.markdown(html_page, unsafe_allow_html=True)

    html2_page = """
    <div style="padding=10px">
        <p style='text-align:center;font-size:20px;font-weight:bold'>Data Science</p>
    </div>
              """
    st.markdown(html2_page, unsafe_allow_html=True)
    
    
     
    image = Image.open("digits.png")
    st.sidebar.image(image,caption="", use_column_width=True)

    activities = ["Home","Digits","PCA","Number of Components","Noisy","PCA as Noise Filtering","About"]
    choice = st.sidebar.selectbox("Home",activities)
    # Using digits read from datasets
    digits = read_digits()
     
    if choice == 'Home':
        st.subheader("Principal Component Analysis")
        st.write("Definitions")
        st.write("PCA is not a statistical method to infer parameters or test hypotheses. Instead, it provides a method to reduce a complex dataset to lower dimension to reveal sometimes hidden, simplified structure that often underlie it.")
        st.write("")
        st.write("PCA is a statistical method routinely used to analyze interrelationships among large numbers of objects.")
        st.write("")
        st.write("Principal component analysis (PCA) is a mathematical algorithm that reduces the dimensionality of the data while retaining most of the variation in the data set.")
        
        if st.button("visit page"):
            js = "window.open('https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html')"
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)
        
        image1 = Image.open("pca.png")
        st.image(image1,caption="", use_column_width=True)

    elif choice == "Digits":
        st.subheader("Demonstration")
        st.write("Shape:",digits.data.shape)
        st.write("Dimensions:",digits.data.shape[1])
        plot_digits(digits)
        
    elif choice == "PCA":
        st.subheader("Applying PCA")
        # Appplying pca to transform in 2 dimensions
        pca = PCA(2)
        project = pca.fit_transform(digits.data)
        st.write("From 64 to 2 dimensions using PCA...")
        st.write("Shape:",project.shape)
        plot_pca(project,digits)

    elif choice == "Number of Components":
        st.subheader("Number of components")
        pca = PCA().fit(digits.data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        st.pyplot()
        st.write("This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components. For example, we see that with the digits the first 10 components contain approximately 75% of the variance, while you need around 50 components to describe close to 100% of the variance.")
        st.write("Using only two-dimensional projection it loses a lot of information (as measured by the explained variance) and that we'd need about 20 components to retain 90% of the variance.")

    elif choice == "Noisy":
        st.subheader("Noisy")
        st.subheader("Original dataset digits")
        #st.write("Shape:",digits.data.shape)
        #st.write("Dimensions:",digits.data.shape[1])
        plot_digits(digits)

        noisy = np.random.normal(digits.data,4)
        st.subheader("Dataset digits with noisy")
        #st.write("Shape:",noisy.shape)
        #st.write("Dimensions:",noisy.shape[1])
        plot_noisy(noisy,digits)
   
        pca = PCA(0.5).fit(noisy)
        
     
    elif choice == "PCA as Noise Filtering":
        st.subheader("PCA as Noise Filtering")
        st.write("Dataset original")
        plot_digits(digits)
        st.write("Dataset with noisy")
        noisy = np.random.normal(digits.data,5)
        plot_noisy(noisy,digits)
        pca = PCA(0.5).fit(noisy)
        components = pca.transform(noisy)
        filtered = pca.inverse_transform(components)
        st.write("Dataset after PCA applyed as noise filtering")
        plot_noisy(filtered,digits)

    elif choice == 'About':
        st.subheader("Built with Streamlit")
        st.subheader("by Silvio Lima")
        
        if st.button("Linkedin"):
            js = "window.open('https://www.linkedin.com/in/silviocesarlima/')"
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)
   


       


       
     

    
   
    
    
if __name__ == '__main__':
    main()
