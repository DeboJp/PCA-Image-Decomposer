from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.array(np.load(filename)) # Load data as numpy array
    x = x - np.mean(x, axis = 0) # Center the dataset by subtracting the mean
    return x

def get_covariance(dataset):
    numPoints = dataset.shape[0] # Determine the number of data points
    covar = np.dot(np.transpose(dataset), dataset) / (numPoints - 1) # Calculate covariance matrix
    return covar

def get_eig(S, m):
    eigenval, eigenvec = eigh(S, subset_by_index=(len(S)-m, len(S)-1)) # Perform eigendecomposition for top m eigenvalues
    eigenval = np.flip(eigenval) # Reverse the array - descending order
    eigenval1 = np.argsort(eigenval) # Get the indices that would sort eigenvalues
    eigenvec = eigenvec / np.linalg.norm(eigenvec, axis = 0) # Normalize eigenvectors
    eigenvec = eigenvec[:, eigenval1] # Rearrange eigenvectors to align with sorted eigenvalues
    diagMatrix = np.diag(eigenval) # A diagonal matrix 
    return diagMatrix, eigenvec

def get_eig_prop(S, prop):
    eigenval, eigenvec = eigh(S) # Compute all eigenvalues and eigenvectors
    eigenval = np.flip(eigenval)
    eigenval1 = np.argsort(eigenval) 
    eigenvec = eigenvec[:, eigenval1]

    threshold = sum(eigenval) * prop # Calculate the variance threshold
    neweigenval = []
    neweigenvec = []
    for i in range(len(eigenval)):
        if eigenval[i] > threshold: # Select eigenvalues and vectors that are above the threshold
            neweigenval.append(eigenval[i])
            neweigenvec.append(eigenvec[:,i])
    neweigenval = np.flip(neweigenval)
    neweigenvec = np.transpose(np.array(neweigenvec)) # A matrix from the list of new eigenvectors
    diagonal = np.diag(neweigenval) # Aiagonal matrix from new eigenvalues

    return diagonal, neweigenvec

def project_image(image, U):
    dot = np.dot(np.transpose(U), image) # Project the image onto the eigenspace
    projection = np.dot(U, dot) # Reconstruct the image from the eigenspace
    return projection

def display_image(orig, proj):
    orgshape = np.transpose(orig.reshape(64,64)) # Reshape and transpose image
    projshape = np.transpose(proj.reshape(64,64)) 

    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2) # Set up a figure with two subplots

    ax1.set_title("Original") # Titles
    ax2.set_title("Projection") 

    fig.colorbar(ax1.imshow(orgshape, aspect='equal'), ax=ax1) # Colorbars
    fig.colorbar(ax2.imshow(projshape, aspect='equal'), ax=ax2) 
    return fig, ax1, ax2

if __name__ == "__main__":
    x = load_and_center_dataset('Iris_64x64.npy') # Load and center the dataset
    S = get_covariance(x) # Covariance matrix of the dataset
    Lambda, U = get_eig(S, 2) # Get eigenvalues and corresponding eigenvectors
    #Feel free to change the zeros to cycle through different images.
    projection = project_image(x[0], U) # Project the first image onto the new eigenspace 
    display_image(x[0], projection) # Display the original and projected image
    plt.show() # Show the plot
