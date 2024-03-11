# parallel-computing-nvidia-cuda

# 1. Problem Description
![Alt Text](https://www.abhinavsriram.com/assets/gif/seamcarve.gif)

**Smart Image Resizing:**
Seam carving assists in resizing images intelligently without excessively distorting crucial objects within the image. This makes image size reduction more effective while maintaining proportions and balance.

**1.1. Potential and Real-world Applications of Seam-Carving Algorithm**

Integrated into Photo Editing Applications:

- Photo editing applications can integrate the seam carving algorithm to provide users with the flexibility to resize images efficiently.

- Adobe incorporated this algorithm into a feature in Photoshop CS4, known as Content Aware Scaling.

Enhancing Image Viewing Experience on Mobile Devices:

- Seam carving can be used to optimize images for mobile devices, improving the image viewing experience on small screens while preserving quality and detail.

Applications in Social Media and Photo Sharing:

- Seam carving can be useful in optimizing image size before sharing on social media platforms, saving bandwidth and speeding up the image uploading process.

Systematic Application in GPU-based Technology:

- For applications requiring rapid image processing, seam carving can be deployed on GPUs to leverage the parallel computing power of GPUs, especially in projects requiring large-scale image and video processing.


**1.2. Description of Input-Output for the Problem**

**Input:**

Original Image:

An image is input into the algorithm for seam carving. The image can be in color or grayscale format.

Size Parameter: The scaling parameter indicates the degree of size reduction of the image along the horizontal axis.

**Output:**

Resulting Image: The image after seam carving with a new size, scaled along the horizontal axis. The image size has been adjusted based on the input parameter.


**1.3. Implementation Steps**

**Step 1:** Perform Grayscale Conversion

**Step 2:** Compute the energy matrix for the image

**Step 3:** Calculate the cumulative energy table of seam values.

**Step 4:** Find the seam with the lowest energy and remove it.

**Step 5:** Repeat the above steps until the image achieves the desired size according to the input Scale parameter.