import streamlit as st

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
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
    fig, axes = plt.subplots(4,10, figsize=(10, 6), subplot_kw= { 'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace = 0.95, wspace=0.05))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i].reshape(8,8),cmap='binary', interpolation='nearest', clim=(0,16))
        ax.text(0,9, str(digits.target[i]))
    st.pyplot()

def plot_noisy(noisy,digits):
    fig, axes = plt.subplots(4,10, figsize=(10, 6), subplot_kw= { 'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace = 0.95, wspace=0.05))
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

    activities = ["Home","Digits","Number of Components","PCA to Reduce The Number of Dimensions","PCA as Noisy Filtering","About"]
    choice = st.sidebar.selectbox("Menu",activities)
    # Using digits read from datasets
    digits = read_digits()
     
    if choice == 'Home':
        st.subheader("Principal Component Analysis")
        st.markdown("**Definitions**")
        st.write("PCA is not a statistical method to infer parameters or test hypotheses. Instead, it provides a method to reduce a complex dataset to lower dimension to reveal sometimes hidden, simplified structure that often underlie it.")
        st.write("")
        st.write("PCA is a statistical method routinely used to analyze interrelationships among large numbers of objects.")
        st.write("")
        st.write("Principal component analysis (PCA) is a mathematical algorithm that reduces the dimensionality of the data while retaining most of the variation in the data set.")
        
        if st.button("Learn more about it"):
            js = "window.open('https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html')"
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)
        
        image1 = Image.open("pca.png")
        st.image(image1,caption="", use_column_width=True)

    elif choice == "Digits":
        st.subheader("Demonstration")
        st.markdown("Applying **PCA**")
        st.markdown("- **to reduce number of dimension**")
        st.markdown("- **to filter noisy**")
        st.write("Original Shape:",digits.data.shape)
        st.write("Dimensions:",digits.data.shape[1])
        plot_digits(digits)
        
    elif choice == "PCA to Reduce The Number of Dimensions":
        st.subheader("Applying PCA")
        # Appplying pca to transform in 2 dimensions
        pca = PCA(2)
        project = pca.fit_transform(digits.data)
        st.write("")
        st.markdown("From **64** to **2** dimensions using PCA...")
        st.write("Shape:",project.shape)
        plot_pca(project,digits)

    elif choice == "Number of Components":
        st.subheader("Number of components")
        st.markdown("Dataset Digits has **64** dimensions to represent the number from 0 to 9")
        st.markdown("**PCA** can help to reduce it. You can inform the **number of components** or the **%rate of variance** to be preserved in new components")
        passagem = st.sidebar.slider("Choose Number of components",min_value=0, max_value=64, value=10, step=5)
        pca = PCA(passagem).fit(digits.data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        st.pyplot()
        st.write("This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components. For example, we see that with the digits the first 10 components contain approximately 75% of the variance, while you need around 50 components to describe close to 100% of the variance.")
        st.write("Using only two-dimensional projection it loses a lot of information (as measured by the explained variance) and that we'd need about 20 components to retain 90% of the variance.")

    elif choice == "PCA as Noisy Filtering":
        st.subheader("PCA as Noisy Filtering")
        st.markdown("**Original dataset digits**")
        #st.write("Shape:",digits.data.shape)
        #st.write("Dimensions:",digits.data.shape[1])
        plot_digits(digits)
        
        #st.write("Shape:",noisy.shape)
        #st.write("Dimensions:",noisy.shape[1])
        noisy_factor = st.sidebar.slider('Noisy Factor ',min_value=0, max_value=10, value=4, step=1)
        noisy = np.random.normal(digits.data,noisy_factor)
        st.markdown("**Dataset digits with noisy**")
        st.write("Noisy:",noisy_factor)
        plot_noisy(noisy,digits)
        
        variance = st.sidebar.slider('% Rate of variance to retain',min_value=0.1, max_value=0.99, value=0.5, step=0.1)
        pca = PCA(variance).fit(noisy)
        components = pca.transform(noisy)
        filtered = pca.inverse_transform(components)
        st.markdown("**Dataset after PCA applyed as noise filtering**")
        rate = str(100* variance)+"%"
        st.write("Rate of variance:",rate)
        st.write("Number of components used: "+ str(pca.n_components_))
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
