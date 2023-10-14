import os
import pandas as pd
import SimpleITK as sitk
import streamlit as st
import matplotlib.pyplot as plt


def dir_selector(folder_path='/Users/fyr/Downloads/'):
    dirnames = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    selected_folder = st.selectbox('Select a folder', dirnames)
    if selected_folder is None:
        return None
    return os.path.join(folder_path, selected_folder)


def plot_slice(vol, slice_ix):
    fig, ax = plt.subplots()
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    ax.imshow(selected_slice, origin='lower', cmap='gray')
    return fig