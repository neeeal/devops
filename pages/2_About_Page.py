
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(
    page_title="About",
    page_icon="📝",
)

# Discussion of best architectures for classification in deep learning
st.write("""
## Best Architectures for Classification in Deep Learning
""")

# Description
st.write("""
This table provides information on the best architectures for classification in deep learning. It will be used as a benchmarking reference.
""")
# Create a table with architecture names, descriptions, and hyperlinks
architectures = {
    "Architecture": ["Convolutional Neural Networks (CNNs)", "Recurrent Neural Networks (RNNs)", "Transformer-based Models", "Support Vector Machines (SVMs)", "Dense Neural Networks", "Ensemble Methods", "Custom Architectures", "AutoML and NAS", "Pre-trained Models", "Hybrid Architectures"],
    "Description": [
        "Effective for image classification and object recognition tasks.",
        "Ideal for sequential data like NLP and time series analysis.",
        "Revolutionized NLP and applied to various domains.",
        "Effective for structured data classification.",
        "Basic building blocks for deep learning models.",
        "Combine multiple models to improve performance.",
        "Designed to address unique challenges in specific problems.",
        "Automated tools to find optimal architectures.",
        "Leverage pre-trained models for specific tasks.",
        "Combine different architectures for hybrid models."
    ],
    "Learn More": [
        "[Learn More](https://en.wikipedia.org/wiki/Convolutional_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Recurrent_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))",
        "[Learn More](https://en.wikipedia.org/wiki/Support_vector_machine)",
        "[Learn More](https://en.wikipedia.org/wiki/Artificial_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)",
        "[Learn More](https://en.wikipedia.org/wiki/Artificial_neural_network#Customization)",
        "[Learn More](https://en.wikipedia.org/wiki/Automated_machine_learning)",
        "[Learn More](https://huggingface.co/transformers/)",
        "[Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)"
    ]
}

# Create a DataFrame and display it with adjusted column widths
import pandas as pd
df = pd.DataFrame(architectures)
st.dataframe(df, width=700)
